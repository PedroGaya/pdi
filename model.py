import copy
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torchvision.models import resnet18, ResNet18_Weights

RANDOM_SEED = 42
IMAGE_SIZE = 160
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 12
NUM_CLASSES = 8
LABEL_SMOOTHING = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

LABEL_MAP = {"N": 0, "D": 1, "G": 2, "C": 3, "A": 4, "H": 5, "M": 6, "O": 7}

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)


def seed_everything(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(num_classes=NUM_CLASSES):
    m = resnet18(weights=ResNet18_Weights.DEFAULT)
    setattr(m, "fc", nn.Sequential(nn.Dropout(0.3), nn.Linear(512, num_classes)))
    return m


def augment(x):
    b = x.size(0)
    angle = torch.rand(b, device=x.device) * 2 * math.pi
    scale = 1 / (0.9 + 0.2 * torch.rand(b, device=x.device))
    flip = (torch.rand(b, device=x.device) < 0.5).float() * 2 - 1
    cos, sin = torch.cos(angle) * scale, torch.sin(angle) * scale
    theta = torch.stack(
        [
            torch.stack([cos * flip, -sin, torch.zeros_like(cos)], 1),
            torch.stack([sin * flip, cos, torch.zeros_like(cos)], 1),
        ],
        1,
    )
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    return x


class OcularDiseaseTrainer:
    def __init__(self, data_dir, csv_file, preprocess_fn):
        self.data_dir = Path(data_dir)
        self.preprocess_fn = preprocess_fn
        self.device = DEVICE

        raw = pd.read_csv(csv_file)
        df = raw[["filename", "labels"]].dropna().copy()
        df["labels"] = df["labels"].astype(str).str.extract(r"([NDGCAHMO])")[0]
        df = df.dropna().reset_index(drop=True)
        if "patient_id" in raw.columns:
            df = df.merge(raw[["filename", "patient_id"]], on="filename", how="left")
            groups = df["patient_id"].astype(str)
        else:
            groups = df["filename"].str.split("_").str[0]
        y = df["labels"].map(LABEL_MAP).to_numpy()

        outer = StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_SEED)
        trainval_idx, test_idx = next(outer.split(df, y, groups))
        inner = StratifiedGroupKFold(6, shuffle=True, random_state=RANDOM_SEED)
        tr, va = next(inner.split(trainval_idx, y[trainval_idx], groups.iloc[trainval_idx]))
        train_idx, val_idx = trainval_idx[tr], trainval_idx[va]

        self.train_df = df.iloc[train_idx].reset_index(drop=True)
        self.val_df = df.iloc[val_idx].reset_index(drop=True)
        self.test_df = df.iloc[test_idx].reset_index(drop=True)

        self.x_train, self.y_train = self._load(self.train_df)
        self.x_val, self.y_val = self._load(self.val_df)
        self.x_test, self.y_test = self._load(self.test_df)

        counts = np.bincount(self.y_train.numpy(), minlength=NUM_CLASSES).clip(min=1)
        weights = (len(self.y_train) / (NUM_CLASSES * counts)) ** 0.5
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        seed_everything()
        self.model = build_model().to(self.device).to(memory_format=torch.channels_last)
        self.frozen = nn.Sequential(self.model.conv1, self.model.bn1, self.model.layer1)
        for prm in self.frozen.parameters():
            prm.requires_grad = False
        self.criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)
        self.optimizer = torch.optim.AdamW(
            [prm for prm in self.model.parameters() if prm.requires_grad],
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        steps = math.ceil(len(self.y_train) / BATCH_SIZE) * NUM_EPOCHS
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=LEARNING_RATE, total_steps=steps, pct_start=0.15
        )
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        baseline = self.train_df["labels"].value_counts(normalize=True).iloc[0]
        print(
            f"Train: {len(self.train_df)} | Val: {len(self.val_df)} | "
            f"Test: {len(self.test_df)} | Baseline: {baseline:.4f}"
        )

    def _read(self, filename):
        bgr = cv2.imread(str(self.data_dir / filename), cv2.IMREAD_COLOR)
        rgb = self.preprocess_fn(bgr)
        rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        return np.transpose(rgb, (2, 0, 1))

    def _load(self, df):
        with ThreadPoolExecutor() as ex:
            imgs = list(ex.map(self._read, df["filename"]))
        x = torch.from_numpy(np.ascontiguousarray(np.stack(imgs)))
        y = torch.tensor(df["labels"].map(LABEL_MAP).to_numpy(), dtype=torch.long)
        return x, y

    def _prep(self, xb):
        return xb.to(self.device, non_blocking=True).float().div_(255.0)

    def _norm(self, x):
        return ((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)

    @torch.no_grad()
    def _predict(self, x_all, tta=True):
        self.model.eval()
        probs = []
        for i in range(0, len(x_all), BATCH_SIZE * 2):
            x = self._prep(x_all[i : i + BATCH_SIZE * 2])
            views = [x, x.flip(3), x.flip(2), x.flip(2).flip(3)] if tta else [x]
            with torch.autocast(self.device.type, enabled=self.use_amp):
                p = torch.stack(
                    [F.softmax(self.model(self._norm(v)).float(), 1) for v in views]
                ).mean(0)
            probs.append(p.cpu())
        return torch.cat(probs)

    def train(self, epochs=NUM_EPOCHS):
        seed_everything()
        best_f1, best_state = -1.0, copy.deepcopy(self.model.state_dict())
        n = len(self.y_train)
        for epoch in range(1, epochs + 1):
            self.model.train()
            self.frozen.eval()
            perm = torch.randperm(n)
            running_loss = 0.0
            for i in range(0, n, BATCH_SIZE):
                idx = perm[i : i + BATCH_SIZE]
                if len(idx) < 2:
                    continue
                images = augment(self._prep(self.x_train[idx]))
                labels = self.y_train[idx].to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(self.device.type, enabled=self.use_amp):
                    loss = self.criterion(self.model(self._norm(images)), labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                running_loss += loss.item() * len(idx)
            preds = self._predict(self.x_val, tta=False).argmax(1)
            val_f1 = f1_score(self.y_val, preds, average="macro", zero_division=0)
            val_acc = (preds == self.y_val).float().mean().item()
            print(
                f"Epoch {epoch:02d}/{epochs:02d} – loss: {running_loss / n:.4f} "
                f"– val acc: {val_acc:.4f} – val macro-F1: {val_f1:.4f}"
            )
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(best_state)

    def evaluate(self):
        preds = self._predict(self.x_test).argmax(1)
        acc = (preds == self.y_test).float().mean().item()
        macro_f1 = f1_score(self.y_test, preds, average="macro", zero_division=0)
        return {"acc": acc, "macro_f1": macro_f1, "preds": preds.tolist()}
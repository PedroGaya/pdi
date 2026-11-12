import itertools
import shutil
import time
from pathlib import Path

import pandas as pd
import kagglehub

from model import OcularDiseaseTrainer

DATA_ROOT = Path(__file__).resolve().parent / "data"
IMAGES_DIR = DATA_ROOT / "images"
METADATA_CSV = DATA_ROOT / "full_df.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

KAGGLE_HANDLE = "andrewmvd/ocular-disease-recognition-odir5k"
LABEL_COLS = ["N", "D", "G", "C", "A", "H", "M", "O"]


def prepare_data(force=False):
    if not force and IMAGES_DIR.is_dir() and METADATA_CSV.is_file():
        return IMAGES_DIR, METADATA_CSV

    root = Path(kagglehub.dataset_download(KAGGLE_HANDLE))
    src_images = next(p for p in root.rglob("preprocessed_images") if p.is_dir())
    src_csv = next(root.rglob("full_df.csv"))

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for f in src_images.rglob("*"):
        if f.is_file():
            shutil.copy2(f, IMAGES_DIR / f.name)

    df = pd.read_csv(src_csv)
    df["filename"] = df["filename"].astype(str).apply(lambda x: Path(x).name)

    def _derive(row):
        codes = []
        for c in LABEL_COLS:
            try:
                if int(float(row[c])) == 1:
                    codes.append(c)
            except (TypeError, ValueError):
                pass
        return ";".join(codes) or "N"

    df["labels"] = df.apply(_derive, axis=1)
    df.to_csv(METADATA_CSV, index=False)
    return IMAGES_DIR, METADATA_CSV


class Orchestrator:
    def __init__(self, out_dir=None):
        self.images_dir, self.csv_file = prepare_data()
        self.out_dir = Path(out_dir) if out_dir else RESULTS_DIR
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _grid(self, grid):
        if not grid:
            yield {}
            return
        keys = list(grid)
        for combo in itertools.product(*(grid[k] for k in keys)):
            yield dict(zip(keys, combo))

    def _sweep(self, name, fn, grid):
        rows = []
        for params in self._grid(grid):
            suffix = "_".join(f"{k}{v}" for k, v in params.items())
            label = f"{name}_{suffix}" if suffix else name

            print(f"\n=== {label} ===")
            t0 = time.time()
            trainer = OcularDiseaseTrainer(
                str(self.images_dir),
                str(self.csv_file),
                lambda bgr, f=fn, p=params: f(bgr, **p),
            )
            trainer.train()
            r = trainer.evaluate()
            elapsed = time.time() - t0

            row = {
                "pipeline": name,
                "params": params,
                "acc": r["acc"],
                "macro_f1": r["macro_f1"],
                "seconds": round(elapsed, 1),
            }
            rows.append(row)
            pd.DataFrame([row]).to_csv(self.out_dir / f"{label}.csv", index=False)

        return rows

    def run(self, experiments):
        rows = []
        for name, (fn, grid) in experiments.items():
            rows += self._sweep(name, fn, grid)
        return rows
import argparse
import csv
import gc
import importlib
import inspect
import itertools
import json
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from model import LABEL_MAP, NUM_CLASSES, OcularDiseaseTrainer

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
IMAGES_DIR = DATA_ROOT / "images"
METADATA_CSV = DATA_ROOT / "full_df.csv"
KAGGLE_HANDLE = "andrewmvd/ocular-disease-recognition-odir5k"
LABEL_COLS = list(LABEL_MAP)

IGNORED_KEYWORDS = (
    "no fundus image",
    "anterior segment image",
    "low image quality",
    "image offset",
    "lens dust",
    "optic disk photographically invisible",
)
KEYWORD_RULES = [
    ("N", ("normal fundus",)),
    ("H", ("hypertensive",)),
    ("M", ("myopia", "myopic")),
    ("O", ("chorioretinopathy", "central serous")),
    ("A", ("macular degeneration",)),
    ("G", ("glaucoma",)),
    ("C", ("cataract",)),
    ("D", ("retinopathy",)),
]
COLUMNS = [
    "timestamp",
    "pipeline",
    "params",
    "acc",
    "macro_f1",
    *[f"f1_{c}" for c in LABEL_MAP],
    "seconds",
]


def _keyword_code(keyword: str) -> str:
    for code, keys in KEYWORD_RULES:
        if any(k in keyword for k in keys):
            return code
    return "O"


def _keywords(text) -> list[str]:
    parts = (k.strip() for k in re.split(r"[,，;；]", str(text).lower()))
    return [k for k in parts if k and not any(i in k for i in IGNORED_KEYWORDS)]


def _find_column(df: pd.DataFrame, side: str) -> str:
    return next(c for c in df.columns if side in c.lower() and "diagnos" in c.lower())


def prepare_data(force: bool = False):
    if not force and IMAGES_DIR.is_dir() and METADATA_CSV.is_file():
        if "patient_id" in pd.read_csv(METADATA_CSV, nrows=0).columns:
            n = sum(1 for _ in IMAGES_DIR.iterdir())
            print(f"Using existing data: {IMAGES_DIR} ({n} files), {METADATA_CSV}")
            return IMAGES_DIR, METADATA_CSV

    import kagglehub

    root = Path(kagglehub.dataset_download(KAGGLE_HANDLE))
    src_images = next(p for p in root.rglob("preprocessed_images") if p.is_dir())
    src_csv = next(root.rglob("full_df.csv"))

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for f in src_images.rglob("*"):
        if f.is_file():
            shutil.copy2(f, IMAGES_DIR / f.name)

    df = pd.read_csv(src_csv)
    filenames = [Path(str(x)).name for x in df["filename"]]
    left_texts = df[_find_column(df, "left")].tolist()
    right_texts = df[_find_column(df, "right")].tolist()
    texts = [
        left if "left" in name.lower() else right
        for name, left, right in zip(filenames, left_texts, right_texts)
    ]

    keywords = [_keywords(t) for t in texts]
    codes = [{_keyword_code(k) for k in ks} for ks in keywords]
    keep = [bool(c) for c in codes]

    flags = df[LABEL_COLS].apply(pd.to_numeric, errors="coerce").fillna(0).eq(1).to_numpy()
    patient_codes = [{c for c, flag in zip(LABEL_COLS, row) if flag} for row in flags]
    mismatches = sum(
        1 for k, e, p in zip(keep, codes, patient_codes) if k and (e - {"N"}) - p
    )
    fallback = Counter(k for ks in keywords for k in ks if _keyword_code(k) == "O")

    ids = (
        [str(x) for x in df["ID"]]
        if "ID" in df.columns
        else [name.split("_")[0] for name in filenames]
    )
    rows = [
        (pid, name, ";".join(c for c in LABEL_COLS if c in cs))
        for pid, name, cs, k in zip(ids, filenames, codes, keep)
        if k
    ]
    out = pd.DataFrame(rows, columns=["patient_id", "filename", "labels"])
    METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(METADATA_CSV, index=False)

    first = Counter(r[2].split(";")[0] for r in rows)
    print(f"Wrote {len(rows)} rows to {METADATA_CSV}")
    return IMAGES_DIR, METADATA_CSV


class Orchestrator:
    def __init__(self, data_dir, csv_file, out_dir=ROOT / "out"):
        self.data_dir = str(data_dir)
        self.csv_file = str(csv_file)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate(fn: Callable[..., np.ndarray], params: dict[str, Any]):
        accepted = list(inspect.signature(fn).parameters)[1:]
        unknown = sorted(set(params) - set(accepted))
        if unknown:
            raise ValueError(
                f"{fn.__name__} does not accept {unknown}; accepted: {accepted}"
            )

    def _append(self, name: str, row: dict[str, Any]):
        path = self.out_dir / f"{name}.csv"
        is_new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if is_new:
                writer.writeheader()
            writer.writerow(row)

    def run(
        self,
        fn: Callable[..., np.ndarray],
        params: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        params = params or {}
        name = name or fn.__name__
        self._validate(fn, params)

        start = time.perf_counter()
        trainer = OcularDiseaseTrainer(self.data_dir, self.csv_file, partial(fn, **params))
        trainer.train()
        result = trainer.evaluate()
        per_class = f1_score(
            trainer.y_test,
            result["preds"],
            labels=list(range(NUM_CLASSES)),
            average=None,
            zero_division=0,
        )
        row: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "pipeline": name,
            "params": json.dumps(params, sort_keys=True),
            "acc": round(float(result["acc"]), 6),
            "macro_f1": round(float(result["macro_f1"]), 6),
            "seconds": round(time.perf_counter() - start, 1),
        }
        for code, score in zip(LABEL_MAP, per_class): # type: ignore
            row[f"f1_{code}"] = round(float(score), 6)

        self._append(name, row)
        del trainer
        gc.collect()
        return row

    def sweep(
        self,
        fn: Callable[..., np.ndarray],
        grid: dict[str, list[Any]] | None = None,
        base: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> list[dict[str, Any]]:
        grid = grid or {}
        base = base or {}
        name = name or fn.__name__
        keys = list(grid)
        combos = list(itertools.product(*grid.values()))
        rows = []
        for i, combo in enumerate(combos, 1):
            params = {**base, **dict(zip(keys, combo))}
            print(f"\n=== [{i}/{len(combos)}] {name} {params} ===")
            row = self.run(fn, params, name)
            print(f"acc={row['acc']:.4f} macro_f1={row['macro_f1']:.4f}")
            rows.append(row)
        print(f"\nResults: {self.out_dir / f'{name}.csv'}")
        return rows


def load_pipeline(spec: str) -> Callable[..., np.ndarray]:
    module_name, _, func_name = spec.partition(":")
    if not func_name:
        raise ValueError("pipeline must be given as module:function")
    sys.path.insert(0, str(Path.cwd()))
    return getattr(importlib.import_module(module_name), func_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline", help="module:function, e.g. pipelines:clahe")
    parser.add_argument("--params", default="{}", help='fixed params as JSON, e.g. \'{"tile": 8}\'')
    parser.add_argument("--grid", default="{}", help='swept params as JSON, e.g. \'{"clip_limit": [1, 2, 4]}\'')
    parser.add_argument("--data-dir")
    parser.add_argument("--csv")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "out"))
    args = parser.parse_args()

    if args.data_dir and args.csv:
        data_dir, csv_file = args.data_dir, args.csv
    else:
        data_dir, csv_file = prepare_data(args.refresh_data)

    Orchestrator(data_dir, csv_file, args.out).sweep(
        load_pipeline(args.pipeline),
        grid=json.loads(args.grid),
        base=json.loads(args.params),
    )


if __name__ == "__main__":
    main()
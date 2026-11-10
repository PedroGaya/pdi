import sys
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd

from orchestrator import Orchestrator, prepare_data


def identity(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def grayscale(bgr):
    return cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)

def clahe(bgr, clip_limit=2.0, tile=8):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

def gaussian_blur(bgr, ksize=5):
    return cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (ksize, ksize), 0)


EXPERIMENTS: dict[str, tuple[Callable[..., np.ndarray], dict[str, list[Any]]]] = {
    "identity": (identity, {}),
    "grayscale": (grayscale, {}),
    "clahe": (clahe, {"clip_limit": [1.0, 2.0, 4.0], "tile": [8]}),
    "gaussian_blur": (gaussian_blur, {"ksize": [3, 5, 9]}),
}


def main():
    selected = sys.argv[1:] or list(EXPERIMENTS)
    unknown = [name for name in selected if name not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"Unknown experiments {unknown}; choose from {list(EXPERIMENTS)}")

    orch = Orchestrator(*prepare_data())
    rows: list[dict[str, Any]] = []
    for name in selected:
        fn, grid = EXPERIMENTS[name]
        rows += orch.sweep(fn, grid)

    summary = pd.DataFrame(rows)[["pipeline", "params", "acc", "macro_f1", "seconds"]]
    print("\n=== Summary ===")
    print(summary.sort_values("macro_f1", ascending=False).to_string(index=False))
    print(f"\nCSV files: {orch.out_dir}")


if __name__ == "__main__":
    main()
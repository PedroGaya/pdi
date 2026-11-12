import cv2
import pandas as pd

from orchestrator import Orchestrator

def clahe(bgr, clip_limit=2.0, tile=8):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

EXPERIMENTS = {
    "clahe": (clahe, {"clip_limit": [2.0, 4.0], "tile": [8.0, 12.0]}),
}

def main():
    orch = Orchestrator()
    rows = orch.run(EXPERIMENTS)
    summary = pd.DataFrame(rows)[["pipeline", "params", "acc", "macro_f1", "seconds"]]
    print("\n=== Summary ===")
    print(summary.sort_values("macro_f1", ascending=False).to_string(index=False))
    print(f"\nCSV files: {orch.out_dir}")

if __name__ == "__main__":
    main()
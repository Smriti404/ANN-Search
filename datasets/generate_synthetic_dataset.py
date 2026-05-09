from __future__ import annotations

"""Generate a synthetic CSV dataset in the format expected by this repo.

Usage:
  python datasets/generate_synthetic_dataset.py --out datasets/my.csv --n 10000 --d 64
"""

from pathlib import Path
import argparse

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    X = rng.normal(size=(args.n, args.d)).astype(np.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    header = ["id"] + [f"v{i}" for i in range(args.d)]
    with out.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for i in range(args.n):
            row = [f"v{i}"] + [f"{x:.6f}" for x in X[i].tolist()]
            f.write(",".join(row) + "\n")

    print(f"Wrote {args.n} vectors (d={args.d}) to {out}")


if __name__ == "__main__":
    main()

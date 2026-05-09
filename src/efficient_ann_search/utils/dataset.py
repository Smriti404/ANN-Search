from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


def load_csv_dataset(path: str | Path) -> Tuple[List[str], np.ndarray]:
    """Load a CSV dataset with format: id,v0,v1,...,v{d-1}."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        if len(header) < 2 or header[0] != "id":
            raise ValueError("CSV must start with header: id,v0,v1,...")
        ids: List[str] = []
        vecs: List[List[float]] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            ids.append(parts[0])
            vecs.append([float(x) for x in parts[1:]])
    return ids, np.array(vecs, dtype=np.float32)

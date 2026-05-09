from __future__ import annotations

import sys
from pathlib import Path


# Enable `src/` layout imports without requiring editable install.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

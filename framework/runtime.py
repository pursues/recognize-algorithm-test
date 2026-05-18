from __future__ import annotations

import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = PROJECT_ROOT / "vendor"

if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

# Keep Ultralytics runtime files inside the project instead of a global user path.
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))

import torch


_RUNTIME_INITIALIZED = False
_ORIGINAL_TORCH_LOAD = torch.load


def _torch_load_compat(*args, **kwargs):
    # Ultralytics 8.0.x expects the pre-PyTorch-2.6 behavior.
    kwargs.setdefault("weights_only", False)
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


def initialize_runtime() -> None:
    global _RUNTIME_INITIALIZED
    if _RUNTIME_INITIALIZED:
        return

    torch.load = _torch_load_compat
    _RUNTIME_INITIALIZED = True


initialize_runtime()

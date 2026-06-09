from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import torch

ROOT_DIR = Path(__file__).resolve().parent  # summary_box 目录

# 加载环境配置文件（必须在其他模块导入前执行）
# 使用非隐藏文件名 env.conf，避免 Jetson 等平台无法读取隐藏文件的问题
_dotenv_path = ROOT_DIR / "env.conf"
if _dotenv_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_dotenv_path)
    except ImportError:
        # 降级方案：手动解析 .env
        with open(_dotenv_path, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _key, _, _val = _line.partition("=")
                _key = _key.strip()
                _val = _val.strip().strip('"').strip("'")
                if _key and _key not in os.environ:
                    os.environ[_key] = _val

PERSON_DIR = ROOT_DIR / "person"
LIBS_DIR = ROOT_DIR / "libs"
TEST_IMAGES_DIR = ROOT_DIR / "imgs"
OUTPUTS_DIR = ROOT_DIR / "outputs"
YOLO_CONFIG_DIR = ROOT_DIR / ".ultralytics"

if LIBS_DIR.exists():
    sys.path.insert(0, str(LIBS_DIR))

os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

try:
    import cv2
except ImportError:
    cv2 = None

DEFAULT_MODEL_NAME = "yolov8n.pt"
DEFAULT_MODEL_PATH = ROOT_DIR / DEFAULT_MODEL_NAME
DEFAULT_CONF = 0.35
DEFAULT_IOU = 0.45
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_RUNTIME_INITIALIZED = False
_ORIGINAL_TORCH_LOAD = torch.load


@dataclass
class Detection:
    label: str
    confidence: float
    xyxy: list[float]
    track_id: int | None = None


@dataclass
class ImagePrediction:
    image_path: Path
    detections: list[Detection]

    @property
    def detected(self) -> bool:
        return bool(self.detections)


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


def initialize_runtime() -> None:
    global _RUNTIME_INITIALIZED
    if _RUNTIME_INITIALIZED:
        return

    torch.load = _torch_load_compat
    YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    _RUNTIME_INITIALIZED = True


def require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("未安装 opencv-python，无法使用摄像头实时识别。")

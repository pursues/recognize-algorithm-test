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

DEFAULT_MODEL_NAME = "best.pt"
DEFAULT_MODEL_PATH = ROOT_DIR / DEFAULT_MODEL_NAME
DEFAULT_CONF = 0.35
DEFAULT_IOU = 0.45
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 行为分类配置：行为名称 → 显示名 + 模型实际标签名（精确匹配）
# 模型实际类别: {0:phones, 1:mask, 2:no_mask, 3:fall, 4:fire, 5:smoke, 6:helmet, 7:smoking}
BEHAVIOR_CONFIG = {
    "fallDown": {
        "display_name": "摔倒检测",
        "expected_labels": ("fall",),
    },
    "fireSmoke": {
        "display_name": "烟雾火灾检测",
        "expected_labels": ("fire", "smoke"),
    },
    "helmet": {
        "display_name": "安全帽检测",
        "expected_labels": ("helmet",),
    },
    "phone": {
        "display_name": "玩手机检测",
        "expected_labels": ("phones",),
    },
    "smoking": {
        "display_name": "抽烟检测",
        "expected_labels": ("smoking",),
    },
    "faceMask": {
        "display_name": "戴口罩检测",
        "expected_labels": ("mask", "no_mask"),
    },
}

def classify_behaviors(detections: list[Detection]) -> dict[str, list[Detection]]:
    """将检测结果按行为类型分类（精确标签匹配）。

    返回: {behavior_name: [匹配的Detection, ...]}
    """
    result: dict[str, list[Detection]] = {}
    for behavior, info in BEHAVIOR_CONFIG.items():
        expected = info["expected_labels"]
        if not expected:
            continue
        matched = []
        for det in detections:
            label_lower = det.label.lower()
            if any(label_lower == exp.lower() for exp in expected):
                matched.append(det)
        if matched:
            result[behavior] = matched
    return result


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

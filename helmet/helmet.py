from __future__ import annotations

from feature_registry import get_feature_config
from framework.detector import Detection
from framework.detector import GenericDetector
from framework.detector import IMAGE_SUFFIXES
from framework.detector import ImagePrediction
from framework.detector import filter_detections
from framework.detector import iter_image_files
from framework.detector import summarize_predictions
from framework.runtime import PROJECT_ROOT


FEATURE_NAME = "helmet"
FEATURE_CONFIG = get_feature_config(FEATURE_NAME)
TEST_IMAGES_DIR = FEATURE_CONFIG.test_images_dir(PROJECT_ROOT)
OUTPUTS_DIR = FEATURE_CONFIG.outputs_dir(PROJECT_ROOT)
MODELS_DIR = FEATURE_CONFIG.models_dir(PROJECT_ROOT)
DEFAULT_HF_MODEL_NAME = "keremberke/yolov8n-hard-hat-detection"
DEFAULT_MODEL_PATH = FEATURE_CONFIG.default_model_path(PROJECT_ROOT)
DEFAULT_MODEL_NAME = str(DEFAULT_MODEL_PATH)

__all__ = [
    "DEFAULT_HF_MODEL_NAME",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_MODEL_PATH",
    "Detection",
    "FEATURE_CONFIG",
    "FEATURE_NAME",
    "HelmetDetector",
    "IMAGE_SUFFIXES",
    "ImagePrediction",
    "MODELS_DIR",
    "OUTPUTS_DIR",
    "TEST_IMAGES_DIR",
    "filter_detections",
    "iter_image_files",
    "summarize_predictions",
]


class HelmetDetector(GenericDetector):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        conf: float | None = None,
        iou: float | None = None,
        agnostic_nms: bool = False,
        max_det: int = 1000,
    ) -> None:
        super().__init__(
            feature_config=FEATURE_CONFIG,
            model_name=model_name,
            conf=conf,
            iou=iou,
            agnostic_nms=agnostic_nms,
            max_det=max_det,
        )

from __future__ import annotations

from dataclasses import dataclass

FEATURE_NAME = "person"
DISPLAY_NAME = "人员识别模型"
EXPECTED_LABELS = ("person",)

# 闯入区域基准值 (基于 1280×720 画面计算，实际使用时根据帧分辨率等比例缩放)
# 在 1280×720 画面中，x: 400~880 为中心水平区域，y: 160~560 为中心垂直区域
_BASE_ZONE_WIDTH = 1280
_BASE_ZONE_HEIGHT = 720
_BASE_ZONE = (400, 160, 880, 560)  # x1, y1, x2, y2

# 闯入重叠阈值：检测框与闯入区域的交集面积占检测框面积的比例
# 默认 0.5，即人的检测框有 50% 以上在红线区域内才算闯入
_INTRUSION_OVERLAP_THRESHOLD = 0.5


@dataclass(frozen=True)
class PersonConfig:
    name: str = FEATURE_NAME
    display_name: str = DISPLAY_NAME
    expected_labels: tuple[str, ...] = EXPECTED_LABELS
    default_conf: float = 0.35
    default_iou: float = 0.45


FEATURE_CONFIG = PersonConfig()

__all__ = [
    "FEATURE_CONFIG",
    "FEATURE_NAME",
    "DISPLAY_NAME",
    "EXPECTED_LABELS",
    "PersonConfig",
    "get_intrusion_zone",
    "is_intrusion",
]

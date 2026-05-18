from __future__ import annotations

from framework.config import FeatureConfig


FEATURE_CONFIGS: dict[str, FeatureConfig] = {
    "helmet": FeatureConfig(
        name="helmet",
        display_name="安全帽检测模型",
        expected_labels=("Hardhat", "NO-Hardhat"),
        default_conf=0.25,
        default_iou=0.45,
        min_positive_hit_rate=0.80,
        max_negative_false_rate=0.20,
    ),
}


def get_feature_config(feature_name: str) -> FeatureConfig:
    try:
        return FEATURE_CONFIGS[feature_name]
    except KeyError as exc:
        supported = ", ".join(sorted(FEATURE_CONFIGS))
        raise ValueError(f"未知功能标识: {feature_name}，当前支持: {supported}") from exc

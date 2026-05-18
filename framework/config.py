from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    display_name: str
    model_file_name: str = "best.pt"
    expected_labels: tuple[str, ...] = ()
    default_conf: float = 0.25
    default_iou: float = 0.45
    min_positive_hit_rate: float = 0.80
    max_negative_false_rate: float = 0.20

    def test_images_dir(self, project_root: Path) -> Path:
        return project_root / "test_images" / self.name

    def outputs_dir(self, project_root: Path) -> Path:
        return project_root / "outputs" / self.name

    def models_dir(self, project_root: Path) -> Path:
        return project_root / "models" / self.name

    def default_model_path(self, project_root: Path) -> Path:
        return self.models_dir(project_root) / self.model_file_name

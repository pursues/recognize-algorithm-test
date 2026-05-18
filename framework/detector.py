from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from typing import Sequence

from framework.runtime import PROJECT_ROOT
from framework.runtime import initialize_runtime

initialize_runtime()

from ultralyticsplus import YOLO

from framework.config import FeatureConfig


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Detection:
    label: str
    confidence: float
    xyxy: list[float]


@dataclass
class ImagePrediction:
    image_path: Path
    detections: list[Detection]

    @property
    def detected(self) -> bool:
        return bool(self.detections)


class GenericDetector:
    def __init__(
        self,
        feature_config: FeatureConfig,
        model_name: str | None = None,
        conf: float | None = None,
        iou: float | None = None,
        agnostic_nms: bool = False,
        max_det: int = 1000,
    ) -> None:
        self.feature_config = feature_config
        self.model_path = (
            Path(model_name)
            if model_name
            else feature_config.default_model_path(PROJECT_ROOT)
        )
        if not self.model_path.exists():
            raise FileNotFoundError(f"默认模型文件不存在: {self.model_path}")

        self.model = YOLO(str(self.model_path))
        self.model.overrides["conf"] = (
            feature_config.default_conf if conf is None else conf
        )
        self.model.overrides["iou"] = feature_config.default_iou if iou is None else iou
        self.model.overrides["agnostic_nms"] = agnostic_nms
        self.model.overrides["max_det"] = max_det

    @property
    def class_names(self) -> dict[int, str]:
        return self.model.model.names

    def predict_image(
        self,
        image_path: str | Path,
        save: bool = False,
        project: str | Path | None = None,
        name: str = "predict",
    ) -> ImagePrediction:
        image_path = Path(image_path)
        predict_kwargs = {
            "source": str(image_path),
            "save": save,
            "name": name,
            "exist_ok": True,
        }
        if project is not None:
            predict_kwargs["project"] = str(project)

        results = self.model.predict(**predict_kwargs)
        result = results[0]
        boxes = result.boxes
        detections: list[Detection] = []

        if boxes is not None:
            for index in range(len(boxes)):
                cls_id = int(boxes.cls[index].item())
                confidence = float(boxes.conf[index].item())
                xyxy = boxes.xyxy[index].tolist()
                detections.append(
                    Detection(
                        label=self.class_names.get(cls_id, str(cls_id)),
                        confidence=confidence,
                        xyxy=xyxy,
                    )
                )

        return ImagePrediction(image_path=image_path, detections=detections)

    def predict_directory(
        self,
        image_dir: str | Path,
        save: bool = False,
        project: str | Path | None = None,
        name: str = "predict",
    ) -> list[ImagePrediction]:
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"图片目录不存在: {image_dir}")

        predictions: list[ImagePrediction] = []
        for image_path in iter_image_files(image_dir):
            predictions.append(
                self.predict_image(
                    image_path=image_path,
                    save=save,
                    project=project,
                    name=name,
                )
            )
        return predictions

    def validate(self, data: str | Path, split: str = "test"):
        return self.model.val(data=str(data), split=split)


def iter_image_files(image_dir: str | Path) -> Iterable[Path]:
    image_dir = Path(image_dir)
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def filter_detections(
    predictions: Sequence[ImagePrediction],
    expected_labels: Sequence[str] | None = None,
) -> list[ImagePrediction]:
    if not expected_labels:
        return list(predictions)

    expected_label_set = set(expected_labels)
    filtered: list[ImagePrediction] = []

    for prediction in predictions:
        filtered_detections = [
            detection
            for detection in prediction.detections
            if detection.label in expected_label_set
        ]
        filtered.append(
            ImagePrediction(
                image_path=prediction.image_path,
                detections=filtered_detections,
            )
        )

    return filtered


def summarize_predictions(predictions: Sequence[ImagePrediction]) -> dict[str, int | float]:
    total_images = len(predictions)
    detected_images = sum(1 for prediction in predictions if prediction.detected)
    total_detections = sum(len(prediction.detections) for prediction in predictions)
    avg_detections = total_detections / total_images if total_images else 0.0

    return {
        "total_images": total_images,
        "detected_images": detected_images,
        "undetected_images": total_images - detected_images,
        "total_detections": total_detections,
        "avg_detections_per_image": round(avg_detections, 4),
    }

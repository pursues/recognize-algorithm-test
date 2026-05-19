from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Sequence

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / ".ultralytics"))

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local environment
    cv2 = None

import torch
from ultralytics import YOLO

PLAY_PHONE_DIR = Path(__file__).resolve().parent
TEST_IMAGES_DIR = PLAY_PHONE_DIR / "imgs"
OUTPUTS_DIR = PLAY_PHONE_DIR / "outputs"
YOLO_CONFIG_DIR = PLAY_PHONE_DIR / ".ultralytics"

FEATURE_NAME = "play_phone"
DISPLAY_NAME = "玩手机识别"
EXPECTED_LABELS = ("cell phone",)
DEFAULT_MODEL_NAME = "yolov8n.pt"
DEFAULT_IMAGE_PATH = TEST_IMAGES_DIR / "phone-game.jpg"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PHONE_LABELS = {"cell phone", "cellphone", "mobile phone", "phone"}
PERSON_LABELS = {"person"}

_RUNTIME_INITIALIZED = False
_ORIGINAL_TORCH_LOAD = torch.load


@dataclass(frozen=True)
class PlayPhoneConfig:
    name: str = FEATURE_NAME
    display_name: str = DISPLAY_NAME
    expected_labels: tuple[str, ...] = EXPECTED_LABELS
    default_conf: float = DEFAULT_CONF
    default_iou: float = DEFAULT_IOU


@dataclass
class Detection:
    label: str
    confidence: float
    xyxy: list[float]


@dataclass
class PhoneUsageEvent:
    person_xyxy: list[float]
    phone_xyxy: list[float]
    face_xyxy: list[float] | None
    score: float
    holding_phone: bool
    looking_phone: bool
    matched_rules: list[str]


@dataclass
class ImagePrediction:
    image_path: Path
    detections: list[Detection]
    face_boxes: list[list[float]]
    phone_usage_events: list[PhoneUsageEvent]

    @property
    def detected(self) -> bool:
        return bool(self.phone_usage_events)


FEATURE_CONFIG = PlayPhoneConfig()

__all__ = [
    "DEFAULT_CONF",
    "DEFAULT_IMAGE_PATH",
    "DEFAULT_IOU",
    "DEFAULT_MODEL_NAME",
    "Detection",
    "DISPLAY_NAME",
    "EXPECTED_LABELS",
    "FEATURE_CONFIG",
    "FEATURE_NAME",
    "IMAGE_SUFFIXES",
    "ImagePrediction",
    "OUTPUTS_DIR",
    "PLAY_PHONE_DIR",
    "PlayPhoneConfig",
    "PlayPhoneDetector",
    "PhoneUsageEvent",
    "TEST_IMAGES_DIR",
    "annotate_prediction",
    "iter_image_files",
    "save_prediction_visualization",
    "summarize_predictions",
]


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


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("未安装 opencv-python，无法进行图片读取与结果可视化。")


def _load_image(image_path: Path):
    _require_cv2()
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"无法读取图片: {image_path}")
    return image


def _build_face_detector():
    if cv2 is None:
        return None
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        return None
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        return None
    return cascade


def _detect_faces(face_detector: Any, image: Any) -> list[list[float]]:
    if face_detector is None or cv2 is None:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(24, 24),
    )
    return [[float(x), float(y), float(x + w), float(y + h)] for x, y, w, h in faces]


def _box_area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _box_center(box: Sequence[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _contains(box: Sequence[float], point: tuple[float, float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _intersection_area(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    if right <= left or bottom <= top:
        return 0.0
    return (right - left) * (bottom - top)


def _find_face_in_person(face_boxes: Sequence[Sequence[float]], person_box: Sequence[float]) -> list[float] | None:
    person_center = _box_center(person_box)
    best_face = None
    best_distance = float("inf")
    for face_box in face_boxes:
        face_center = _box_center(face_box)
        if not _contains(person_box, face_center):
            continue
        distance = math.dist(person_center, face_center)
        if distance < best_distance:
            best_distance = distance
            best_face = list(face_box)
    return best_face


def infer_phone_usage(
    detections: Sequence[Detection],
    face_boxes: Sequence[Sequence[float]],
) -> list[PhoneUsageEvent]:
    people = [detection for detection in detections if detection.label in PERSON_LABELS]
    phones = [detection for detection in detections if detection.label in PHONE_LABELS]
    events: list[PhoneUsageEvent] = []

    for person in people:
        person_box = person.xyxy
        person_area = _box_area(person_box)
        person_width = max(person_box[2] - person_box[0], 1.0)
        person_height = max(person_box[3] - person_box[1], 1.0)
        person_center = _box_center(person_box)
        person_upper_limit = person_box[1] + person_height * 0.78
        face_box = _find_face_in_person(face_boxes, person_box)

        for phone in phones:
            phone_box = phone.xyxy
            phone_center = _box_center(phone_box)
            phone_area = _box_area(phone_box)
            phone_area_ratio = phone_area / max(person_area, 1.0)
            phone_overlap_ratio = _intersection_area(person_box, phone_box) / max(phone_area, 1.0)
            phone_inside_person = _contains(person_box, phone_center) or phone_overlap_ratio >= 0.6
            phone_in_upper_body = phone_center[1] <= person_upper_limit
            phone_near_body_center = abs(phone_center[0] - person_center[0]) / person_width <= 0.45
            reasonable_phone_size = 0.002 <= phone_area_ratio <= 0.18

            matched_rules: list[str] = []
            score = 0.0

            if phone_inside_person:
                score += 0.4
                matched_rules.append("phone_inside_person")
            if phone_in_upper_body:
                score += 0.15
                matched_rules.append("phone_in_upper_body")
            if phone_near_body_center:
                score += 0.1
                matched_rules.append("phone_near_body_center")
            if reasonable_phone_size:
                score += 0.15
                matched_rules.append("reasonable_phone_size")

            holding_phone = (
                phone_inside_person
                and reasonable_phone_size
                and (phone_in_upper_body or phone_near_body_center)
            )
            if holding_phone:
                score += 0.1
                matched_rules.append("holding_phone")

            looking_phone = False
            if face_box is not None:
                face_center = _box_center(face_box)
                face_width = max(face_box[2] - face_box[0], 1.0)
                face_height = max(face_box[3] - face_box[1], 1.0)
                phone_below_face = phone_center[1] >= face_center[1] + face_height * 0.1
                phone_close_to_face = math.dist(phone_center, face_center) / math.hypot(
                    person_width,
                    person_height,
                ) <= 0.45
                horizontally_aligned = abs(phone_center[0] - face_center[0]) / person_width <= 0.35
                phone_wider_than_face_half = (phone_box[2] - phone_box[0]) >= face_width * 0.5

                if phone_below_face:
                    score += 0.1
                    matched_rules.append("phone_below_face")
                if phone_close_to_face:
                    score += 0.1
                    matched_rules.append("phone_close_to_face")
                if horizontally_aligned:
                    score += 0.1
                    matched_rules.append("phone_horizontally_aligned")
                if phone_wider_than_face_half:
                    score += 0.05
                    matched_rules.append("phone_wider_than_face_half")

                looking_phone = phone_below_face and phone_close_to_face and horizontally_aligned
            else:
                if phone_center[1] <= person_box[1] + person_height * 0.68:
                    score += 0.08
                    matched_rules.append("phone_high_without_face")
                if abs(phone_center[0] - person_center[0]) / person_width <= 0.3:
                    score += 0.07
                    matched_rules.append("phone_centered_without_face")
                looking_phone = (
                    phone_inside_person
                    and phone_in_upper_body
                    and abs(phone_center[0] - person_center[0]) / person_width <= 0.3
                )

            final_score = round(min(score, 1.0), 4)
            if final_score >= 0.7 and holding_phone:
                events.append(
                    PhoneUsageEvent(
                        person_xyxy=[round(value, 2) for value in person_box],
                        phone_xyxy=[round(value, 2) for value in phone_box],
                        face_xyxy=[round(value, 2) for value in face_box] if face_box else None,
                        score=final_score,
                        holding_phone=holding_phone,
                        looking_phone=looking_phone,
                        matched_rules=matched_rules,
                    )
                )

    return events


class PlayPhoneDetector:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        conf: float | None = None,
        iou: float | None = None,
        agnostic_nms: bool = False,
        max_det: int = 300,
    ) -> None:
        initialize_runtime()

        self.feature_config = FEATURE_CONFIG
        self.model_name = model_name
        model_path = Path(model_name)
        model_source = str(model_path.resolve()) if model_path.exists() else model_name
        self.model = YOLO(model_source)
        self.model.overrides["conf"] = DEFAULT_CONF if conf is None else conf
        self.model.overrides["iou"] = DEFAULT_IOU if iou is None else iou
        self.model.overrides["agnostic_nms"] = agnostic_nms
        self.model.overrides["max_det"] = max_det
        self.face_detector = _build_face_detector()

    @property
    def class_names(self) -> dict[int, str]:
        return self.model.model.names

    def _build_prediction(self, result: Any, image_path: Path, image: Any) -> ImagePrediction:
        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is not None:
            for index in range(len(boxes)):
                cls_id = int(boxes.cls[index].item())
                confidence = float(boxes.conf[index].item())
                xyxy = boxes.xyxy[index].tolist()
                detections.append(
                    Detection(
                        label=self.class_names.get(cls_id, str(cls_id)),
                        confidence=confidence,
                        xyxy=[float(value) for value in xyxy],
                    )
                )

        face_boxes = _detect_faces(self.face_detector, image=image)
        phone_usage_events = infer_phone_usage(detections=detections, face_boxes=face_boxes)
        return ImagePrediction(
            image_path=image_path,
            detections=detections,
            face_boxes=[[round(value, 2) for value in face_box] for face_box in face_boxes],
            phone_usage_events=phone_usage_events,
        )

    def predict_image(self, image_path: str | Path) -> ImagePrediction:
        image_path = Path(image_path)
        image = _load_image(image_path)
        results = self.model.predict(
            source=str(image_path),
            conf=self.model.overrides["conf"],
            iou=self.model.overrides["iou"],
            verbose=False,
        )
        return self._build_prediction(results[0], image_path=image_path, image=image)

    def predict_directory(self, image_dir: str | Path) -> list[ImagePrediction]:
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"图片目录不存在: {image_dir}")

        predictions: list[ImagePrediction] = []
        for image_path in iter_image_files(image_dir):
            predictions.append(self.predict_image(image_path))
        return predictions


def annotate_prediction(prediction: ImagePrediction):
    image = _load_image(prediction.image_path)
    if cv2 is None:
        return image

    for detection in prediction.detections:
        color = (0, 255, 255)
        if detection.label in PERSON_LABELS:
            color = (255, 128, 0)
        elif detection.label in PHONE_LABELS:
            color = (0, 200, 255)
        x1, y1, x2, y2 = [int(round(value)) for value in detection.xyxy]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"{detection.label} {detection.confidence:.2f}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    for face_box in prediction.face_boxes:
        x1, y1, x2, y2 = [int(round(value)) for value in face_box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)

    for event in prediction.phone_usage_events:
        px1, py1, px2, py2 = [int(round(value)) for value in event.person_xyxy]
        fx = px1
        fy = max(30, py1 - 12)
        cv2.rectangle(image, (px1, py1), (px2, py2), (0, 0, 255), 3)
        cv2.putText(
            image,
            f"play_phone score={event.score:.2f}",
            (fx, fy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        phx1, phy1, phx2, phy2 = [int(round(value)) for value in event.phone_xyxy]
        cv2.rectangle(image, (phx1, phy1), (phx2, phy2), (0, 0, 255), 3)

    return image


def save_prediction_visualization(
    prediction: ImagePrediction,
    output_dir: str | Path,
    name: str = "predict",
) -> Path:
    output_dir = Path(output_dir)
    save_dir = output_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)
    annotated = annotate_prediction(prediction)
    save_path = save_dir / prediction.image_path.name
    if cv2 is None:
        raise RuntimeError("未安装 opencv-python，无法保存可视化结果。")
    cv2.imwrite(str(save_path), annotated)
    return save_path


def iter_image_files(image_dir: str | Path) -> Iterable[Path]:
    image_dir = Path(image_dir)
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def summarize_predictions(predictions: Sequence[ImagePrediction]) -> dict[str, int | float]:
    total_images = len(predictions)
    detected_images = sum(1 for prediction in predictions if prediction.detected)
    total_detections = sum(len(prediction.phone_usage_events) for prediction in predictions)
    avg_detections = total_detections / total_images if total_images else 0.0

    return {
        "total_images": total_images,
        "detected_images": detected_images,
        "undetected_images": total_images - detected_images,
        "total_play_phone_events": total_detections,
        "avg_events_per_image": round(avg_detections, 4),
    }

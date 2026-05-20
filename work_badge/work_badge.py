from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
import shutil
from typing import Any
from typing import Iterable
from typing import Sequence
import sys

WORK_BADGE_DIR = Path(__file__).resolve().parent

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local environment
    cv2 = None

from huggingface_hub import snapshot_download
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection
from transformers import AutoProcessor
from ultralytics import YOLO

TEST_IMAGES_DIR = WORK_BADGE_DIR / "imgs"
MODELS_DIR = WORK_BADGE_DIR / "models"
OUTPUTS_DIR = WORK_BADGE_DIR / "outputs"
PROJECT_ROOT = WORK_BADGE_DIR.parent
DEFAULT_PERSON_MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"
DEFAULT_PERSON_MODEL_NAME = str(DEFAULT_PERSON_MODEL_PATH)
DEFAULT_ZERO_SHOT_MODEL_REPO_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_ZERO_SHOT_MODEL_PATH = MODELS_DIR / "grounding-dino-tiny"
DEFAULT_ZERO_SHOT_MODEL_NAME = str(DEFAULT_ZERO_SHOT_MODEL_PATH)
DEFAULT_IMAGE_PATH = TEST_IMAGES_DIR / "badge.png"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_BOX_THRESHOLD = 0.18
DEFAULT_TEXT_THRESHOLD = 0.18
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FEATURE_NAME = "work_badge"
DISPLAY_NAME = "工作牌识别"
DEFAULT_BADGE_QUERIES = (
    "employee badge",
    "work badge",
    "name tag",
    "id card",
    "lanyard badge",
    "identification badge",
)

_RUNTIME_INITIALIZED = False


@dataclass(frozen=True)
class WorkBadgeConfig:
    name: str = FEATURE_NAME
    display_name: str = DISPLAY_NAME
    default_zero_shot_model: str = DEFAULT_ZERO_SHOT_MODEL_NAME
    default_zero_shot_repo_id: str = DEFAULT_ZERO_SHOT_MODEL_REPO_ID
    badge_queries: tuple[str, ...] = DEFAULT_BADGE_QUERIES
    default_conf: float = DEFAULT_CONF
    default_iou: float = DEFAULT_IOU
    default_box_threshold: float = DEFAULT_BOX_THRESHOLD
    default_text_threshold: float = DEFAULT_TEXT_THRESHOLD


@dataclass
class BadgeDetection:
    label: str
    confidence: float
    xyxy: list[float]


@dataclass
class PersonBadgeDetection:
    label: str
    confidence: float
    person_xyxy: list[float]
    chest_xyxy: list[float]
    badge_detections: list[BadgeDetection]
    is_wearing_badge: bool
    status: str


@dataclass
class ImagePrediction:
    image_path: Path
    detections: list[PersonBadgeDetection]

    @property
    def detected(self) -> bool:
        return any(detection.is_wearing_badge for detection in self.detections)


FEATURE_CONFIG = WorkBadgeConfig()

__all__ = [
    "BadgeDetection",
    "DEFAULT_BADGE_QUERIES",
    "DEFAULT_BOX_THRESHOLD",
    "DEFAULT_CONF",
    "DEFAULT_IMAGE_PATH",
    "DEFAULT_IOU",
    "DEFAULT_PERSON_MODEL_NAME",
    "DEFAULT_PERSON_MODEL_PATH",
    "DEFAULT_TEXT_THRESHOLD",
    "DEFAULT_ZERO_SHOT_MODEL_NAME",
    "DEFAULT_ZERO_SHOT_MODEL_PATH",
    "DEFAULT_ZERO_SHOT_MODEL_REPO_ID",
    "DISPLAY_NAME",
    "FEATURE_CONFIG",
    "FEATURE_NAME",
    "IMAGE_SUFFIXES",
    "ImagePrediction",
    "MODELS_DIR",
    "OUTPUTS_DIR",
    "PROJECT_ROOT",
    "PersonBadgeDetection",
    "TEST_IMAGES_DIR",
    "WORK_BADGE_DIR",
    "WorkBadgeConfig",
    "WorkBadgeDetector",
    "annotate_prediction",
    "iter_image_files",
    "save_prediction_visualization",
    "summarize_predictions",
]


def initialize_runtime() -> None:
    global _RUNTIME_INITIALIZED
    if _RUNTIME_INITIALIZED:
        return
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _RUNTIME_INITIALIZED = True


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("未安装 opencv-python，无法生成可视化结果。")


def _load_image(image_path: Path):
    _require_cv2()
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"无法读取图片: {image_path}")
    return image


def _normalize_text(label: str) -> str:
    return label.strip().lower().replace("-", " ").replace("_", " ")


def _clip_box(box: Sequence[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    return [
        max(int(round(x1)), 0),
        max(int(round(y1)), 0),
        min(int(round(x2)), width),
        min(int(round(y2)), height),
    ]


def build_csi_gstreamer_pipeline(
    width: int = 1280,
    height: int = 720,
    framerate: int = 30,
    flip_method: int = 0,
) -> str:
    return (
        "nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink"
    )


def open_camera(
    camera: str | int = "usb",
    width: int = 1280,
    height: int = 720,
    framerate: int = 30,
    flip_method: int = 0,
):
    _require_cv2()
    if isinstance(camera, int):
        return cv2.VideoCapture(camera)
    if camera == "csi":
        gst_str = build_csi_gstreamer_pipeline(
            width=width,
            height=height,
            framerate=framerate,
            flip_method=flip_method,
        )
        return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    if camera == "usb":
        return cv2.VideoCapture(0, cv2.CAP_ANY)
    if camera.isdigit():
        return cv2.VideoCapture(int(camera))
    raise ValueError(f"不支持的摄像头类型: {camera}，可选值为 csi、usb 或数字序号")


def iter_image_files(image_dir: str | Path) -> Iterable[Path]:
    image_dir = Path(image_dir)
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def _copy_snapshot_to_local_dir(snapshot_dir: Path, local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    for child in snapshot_dir.iterdir():
        target = local_dir / child.name
        if child.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)
    return local_dir


def ensure_local_zero_shot_model(
    model_path: str | Path = DEFAULT_ZERO_SHOT_MODEL_PATH,
    repo_id: str = DEFAULT_ZERO_SHOT_MODEL_REPO_ID,
    allow_download: bool = False,
) -> Path:
    initialize_runtime()
    local_dir = Path(model_path)
    if local_dir.exists():
        return local_dir
    if not allow_download:
        raise FileNotFoundError(
            f"本地零样本模型不存在: {local_dir}。"
            f"请先下载模型到该目录，或在初始化时开启 allow_download。"
        )
    from huggingface_hub import snapshot_download
    print(f"Downloading model {repo_id} to {local_dir}...")
    snapshot_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
        )
    )
    if snapshot_dir != local_dir:
        _copy_snapshot_to_local_dir(snapshot_dir, local_dir)
    return local_dir


class ZeroShotBadgeLocalizer:
    def __init__(
        self,
        model_name: str = DEFAULT_ZERO_SHOT_MODEL_NAME,
        repo_id: str = DEFAULT_ZERO_SHOT_MODEL_REPO_ID,
        box_threshold: float = DEFAULT_BOX_THRESHOLD,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
        device: str | None = None,
        allow_download: bool = False,
    ) -> None:
        model_path = ensure_local_zero_shot_model(
            model_path=model_name,
            repo_id=repo_id,
            allow_download=allow_download,
        )
        self.model_name = str(model_path)
        self.repo_id = repo_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            str(model_path),
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        self.model_type = str(getattr(self.model.config, "model_type", "")).lower()
        self._grounding_post_process_params = set(
            inspect.signature(self.processor.post_process_grounded_object_detection).parameters.keys()
        )

    def _prepare_text(self, candidate_labels: Sequence[str]) -> list[list[str]]:
        if "grounding" in self.model_type:
            return [[label for label in candidate_labels]]
        return [[f"a photo of a {label}" for label in candidate_labels]]

    def detect(self, image: Image.Image, candidate_labels: Sequence[str]) -> list[dict[str, object]]:
        text = self._prepare_text(candidate_labels)
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = [image.size[::-1]]
        if "grounding" in self.model_type:
            kwargs: dict[str, object] = {
                "outputs": outputs,
                "input_ids": inputs["input_ids"],
                "text_threshold": self.text_threshold,
                "target_sizes": target_sizes,
            }
            if "box_threshold" in self._grounding_post_process_params:
                kwargs["box_threshold"] = self.box_threshold
            elif "threshold" in self._grounding_post_process_params:
                kwargs["threshold"] = self.box_threshold
            else:
                kwargs["threshold"] = self.box_threshold
            results = self.processor.post_process_grounded_object_detection(**kwargs)[0]
        else:
            results = self.processor.post_process_object_detection(
                outputs=outputs,
                threshold=self.box_threshold,
                target_sizes=target_sizes,
            )[0]

        detections: list[dict[str, object]] = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"], strict=True):
            label_text = label if isinstance(label, str) else text[0][int(label)]
            detections.append(
                {
                    "label": str(label_text),
                    "score": float(score),
                    "box": [float(value) for value in box.tolist()],
                }
            )
        return detections


class WorkBadgeDetector:
    def __init__(
        self,
        person_model_name: str = DEFAULT_PERSON_MODEL_NAME,
        zero_shot_model_name: str = DEFAULT_ZERO_SHOT_MODEL_NAME,
        zero_shot_repo_id: str = DEFAULT_ZERO_SHOT_MODEL_REPO_ID,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
        box_threshold: float = DEFAULT_BOX_THRESHOLD,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
        badge_queries: Sequence[str] = DEFAULT_BADGE_QUERIES,
        allow_download: bool = False,
    ) -> None:
        initialize_runtime()

        self.feature_config = FEATURE_CONFIG
        self.person_model_path = Path(person_model_name)
        if not self.person_model_path.exists():
            raise FileNotFoundError(f"人员检测权重不存在: {self.person_model_path}")

        self.conf = conf
        self.iou = iou
        self.badge_queries = tuple(badge_queries)
        self.person_model = YOLO(str(self.person_model_path))
        self.badge_localizer = ZeroShotBadgeLocalizer(
            model_name=zero_shot_model_name,
            repo_id=zero_shot_repo_id,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            allow_download=allow_download,
        )

    def _build_chest_box(self, person_box: Sequence[float], image_size: tuple[int, int]) -> list[int]:
        image_width, image_height = image_size
        x1, y1, x2, y2 = person_box
        person_width = max(x2 - x1, 1.0)
        person_height = max(y2 - y1, 1.0)
        chest_box = [
            x1 + person_width * 0.12,
            y1 + person_height * 0.12,
            x2 - person_width * 0.12,
            y1 + person_height * 0.82,
        ]
        return _clip_box(chest_box, width=image_width, height=image_height)

    def _filter_badge_detections(
        self,
        raw_detections: Sequence[dict[str, object]],
        chest_box: Sequence[int],
        image_size: tuple[int, int],
    ) -> list[BadgeDetection]:
        image_width, image_height = image_size
        chest_x1, chest_y1, chest_x2, chest_y2 = chest_box
        chest_width = max(chest_x2 - chest_x1, 1)
        chest_height = max(chest_y2 - chest_y1, 1)
        chest_area = chest_width * chest_height

        filtered: list[BadgeDetection] = []
        for item in raw_detections:
            local_box = item["box"]
            label = str(item["label"])
            score = float(item["score"])
            local_x1, local_y1, local_x2, local_y2 = local_box
            global_box = _clip_box(
                [
                    chest_x1 + local_x1,
                    chest_y1 + local_y1,
                    chest_x1 + local_x2,
                    chest_y1 + local_y2,
                ],
                width=image_width,
                height=image_height,
            )
            x1, y1, x2, y2 = global_box
            width = max(x2 - x1, 1)
            height = max(y2 - y1, 1)
            area_ratio = (width * height) / chest_area
            aspect_ratio = width / height
            width_ratio = width / chest_width
            height_ratio = height / chest_height
            center_x_ratio = ((x1 + x2) / 2 - chest_x1) / chest_width
            center_y_ratio = ((y1 + y2) / 2 - chest_y1) / chest_height
            normalized_label = _normalize_text(label)
            label_ok = "badge" in normalized_label or "card" in normalized_label or "tag" in normalized_label
            geometry_ok = (
                0.006 <= area_ratio <= 0.2
                and 0.2 <= aspect_ratio <= 1.2
                and 0.08 <= width_ratio <= 0.35
                and 0.08 <= height_ratio <= 0.35
            )
            position_ok = 0.05 <= center_x_ratio <= 0.95 and 0.45 <= center_y_ratio <= 0.98
            if label_ok and geometry_ok and position_ok and score >= self.badge_localizer.box_threshold:
                filtered.append(
                    BadgeDetection(
                        label=label,
                        confidence=round(score, 4),
                        xyxy=[round(float(value), 2) for value in global_box],
                    )
                )
        return filtered

    def _build_person_detection(
        self,
        person_box: Sequence[float],
        person_confidence: float,
        badge_detections: list[BadgeDetection],
        image_size: tuple[int, int],
    ) -> PersonBadgeDetection:
        person_xyxy = _clip_box(person_box, width=image_size[0], height=image_size[1])
        chest_xyxy = self._build_chest_box(person_xyxy, image_size=image_size)
        is_wearing_badge = bool(badge_detections)
        return PersonBadgeDetection(
            label="person",
            confidence=round(float(person_confidence), 4),
            person_xyxy=[round(float(value), 2) for value in person_xyxy],
            chest_xyxy=[round(float(value), 2) for value in chest_xyxy],
            badge_detections=badge_detections,
            is_wearing_badge=is_wearing_badge,
            status="with_work_badge" if is_wearing_badge else "without_work_badge",
        )

    def _predict_from_pil(
        self,
        pil_image: Image.Image,
        source: str | Path | Any,
        image_path: Path,
    ) -> ImagePrediction:
        image_size = pil_image.size
        result = self.person_model.predict(
            source=source,
            classes=[0],
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )[0]

        detections: list[PersonBadgeDetection] = []
        if result.boxes is not None:
            person_boxes = result.boxes.xyxy.cpu().tolist()
            person_confidences = result.boxes.conf.cpu().tolist()
            for person_box, person_confidence in zip(person_boxes, person_confidences, strict=True):
                chest_box = self._build_chest_box(person_box, image_size=image_size)
                crop = pil_image.crop(tuple(chest_box))
                raw_badges = self.badge_localizer.detect(crop, self.badge_queries)
                badge_detections = self._filter_badge_detections(
                    raw_detections=raw_badges,
                    chest_box=chest_box,
                    image_size=image_size,
                )
                detections.append(
                    self._build_person_detection(
                        person_box=person_box,
                        person_confidence=person_confidence,
                        badge_detections=badge_detections,
                        image_size=image_size,
                    )
                )

        return ImagePrediction(image_path=image_path, detections=detections)

    def predict_image(self, image_path: str | Path) -> ImagePrediction:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图片不存在: {image_path}")
        pil_image = Image.open(image_path).convert("RGB")
        return self._predict_from_pil(
            pil_image=pil_image,
            source=str(image_path),
            image_path=image_path,
        )

    def predict_frame(
        self,
        frame: Any,
        frame_name: str = "camera_frame",
    ) -> ImagePrediction:
        _require_cv2()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        return self._predict_from_pil(
            pil_image=pil_image,
            source=frame,
            image_path=Path(frame_name),
        )

    def predict_directory(self, image_dir: str | Path) -> list[ImagePrediction]:
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"图片目录不存在: {image_dir}")
        return [self.predict_image(image_path) for image_path in iter_image_files(image_dir)]

    def run_camera_inference(
        self,
        camera: str | int = "csi",
        frame_log_interval: int = 10,
        window_name: str = "Work Badge Detection",
        width: int = 1280,
        height: int = 720,
        framerate: int = 30,
        flip_method: int = 0,
        quit_key: str = "q",
        warmup_frames: int = 5,
        max_failed_reads: int = 30,
    ) -> None:
        _require_cv2()
        cap = open_camera(
            camera=camera,
            width=width,
            height=height,
            framerate=framerate,
            flip_method=flip_method,
        )
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头: {camera}")

        print(f"实时识别已启动，摄像头={camera}，按 '{quit_key.upper()}' 退出。")
        frame_count = 0
        failed_reads = 0

        try:
            for _ in range(max(warmup_frames, 0)):
                cap.read()

            while True:
                try:
                    ret, frame = cap.read()
                except Exception as exc:
                    raise RuntimeError(
                        f"读取摄像头帧失败: camera={camera}。请检查摄像头类型、驱动或 GStreamer 配置。"
                    ) from exc

                if not ret:
                    failed_reads += 1
                    if failed_reads >= max_failed_reads:
                        raise RuntimeError(f"连续 {failed_reads} 次读取摄像头帧失败: camera={camera}。")
                    continue
                failed_reads = 0

                frame_count += 1
                prediction = self.predict_frame(frame, frame_name=f"camera:{camera}")
                annotated_frame = _annotate_image(frame.copy(), prediction)

                if frame_log_interval > 0 and frame_count % frame_log_interval == 0:
                    if prediction.detections:
                        for detection in prediction.detections:
                            print(
                                f"[DETECT] {detection.status} person: {detection.confidence:.2f}, "
                                f"badge_count={len(detection.badge_detections)}"
                            )
                    else:
                        print("[DETECT] No person detections in this frame.")

                cv2.imshow(window_name, annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord(quit_key.lower()):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


def _annotate_image(image, prediction: ImagePrediction):
    if cv2 is None:
        return image

    for detection in prediction.detections:
        person_color = (0, 180, 0) if detection.is_wearing_badge else (0, 0, 255)
        chest_color = (255, 140, 0)

        px1, py1, px2, py2 = [int(round(value)) for value in detection.person_xyxy]
        cx1, cy1, cx2, cy2 = [int(round(value)) for value in detection.chest_xyxy]
        cv2.rectangle(image, (px1, py1), (px2, py2), person_color, 2)
        cv2.rectangle(image, (cx1, cy1), (cx2, cy2), chest_color, 1)

        status_text = f"{detection.status} {detection.confidence:.2f}"
        cv2.putText(
            image,
            status_text,
            (px1, max(24, py1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            person_color,
            2,
            cv2.LINE_AA,
        )

        for badge in detection.badge_detections:
            bx1, by1, bx2, by2 = [int(round(value)) for value in badge.xyxy]
            cv2.rectangle(image, (bx1, by1), (bx2, by2), (255, 255, 0), 2)
            badge_text = f"{badge.label} {badge.confidence:.2f}"
            cv2.putText(
                image,
                badge_text,
                (bx1, max(24, by1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

    return image


def annotate_prediction(prediction: ImagePrediction):
    image = _load_image(prediction.image_path)
    return _annotate_image(image, prediction)


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


def summarize_predictions(predictions: Sequence[ImagePrediction]) -> dict[str, int | float]:
    total_images = len(predictions)
    total_persons = sum(len(prediction.detections) for prediction in predictions)
    badge_positive_persons = sum(
        1
        for prediction in predictions
        for detection in prediction.detections
        if detection.is_wearing_badge
    )
    badge_negative_persons = total_persons - badge_positive_persons
    return {
        "total_images": total_images,
        "detected_images": sum(1 for prediction in predictions if prediction.detected),
        "undetected_images": sum(1 for prediction in predictions if not prediction.detected),
        "total_persons": total_persons,
        "badge_positive_persons": badge_positive_persons,
        "badge_negative_persons": badge_negative_persons,
        "avg_persons_per_image": round(total_persons / total_images, 4) if total_images else 0.0,
    }

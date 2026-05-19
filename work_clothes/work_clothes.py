from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Sequence

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local environment
    cv2 = None

from ultralytics import YOLO

WORK_CLOTHES_DIR = Path(__file__).resolve().parent
TEST_IMAGES_DIR = WORK_CLOTHES_DIR / "imgs"
MODELS_DIR = WORK_CLOTHES_DIR / "models"
OUTPUTS_DIR = WORK_CLOTHES_DIR / "outputs"
PROJECT_ROOT = WORK_CLOTHES_DIR.parent
DEFAULT_PERSON_MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"
DEFAULT_PERSON_MODEL_NAME = str(DEFAULT_PERSON_MODEL_PATH)
DEFAULT_PPE_MODEL_PATH = MODELS_DIR / "safety_vest_best.pt"
DEFAULT_PPE_MODEL_NAME = str(DEFAULT_PPE_MODEL_PATH)
DEFAULT_IMAGE_PATH = TEST_IMAGES_DIR / "clothe.png"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FEATURE_NAME = "safety_vest"
DISPLAY_NAME = "反光背心/安全马甲识别"
POSITIVE_LABELS = ("person_with_vest", "Safety Vest", "Vest", "Reflective-Jacket")
NEGATIVE_LABELS = ("No Vest", "NO-Safety Vest")

_RUNTIME_INITIALIZED = False


@dataclass(frozen=True)
class WorkClothesConfig:
    name: str = FEATURE_NAME
    display_name: str = DISPLAY_NAME
    positive_labels: tuple[str, ...] = POSITIVE_LABELS
    negative_labels: tuple[str, ...] = NEGATIVE_LABELS
    default_conf: float = DEFAULT_CONF
    default_iou: float = DEFAULT_IOU


@dataclass
class PersonDetection:
    label: str
    raw_class_name: str
    confidence: float
    xyxy: list[float]
    is_work_clothes: bool
    status: str


@dataclass
class ImagePrediction:
    image_path: Path
    detections: list[PersonDetection]

    @property
    def detected(self) -> bool:
        return any(detection.is_work_clothes for detection in self.detections)


FEATURE_CONFIG = WorkClothesConfig()

__all__ = [
    "DEFAULT_CONF",
    "DEFAULT_IMAGE_PATH",
    "DEFAULT_IOU",
    "DEFAULT_PERSON_MODEL_NAME",
    "DEFAULT_PERSON_MODEL_PATH",
    "DEFAULT_PPE_MODEL_NAME",
    "DEFAULT_PPE_MODEL_PATH",
    "DISPLAY_NAME",
    "FEATURE_CONFIG",
    "FEATURE_NAME",
    "IMAGE_SUFFIXES",
    "ImagePrediction",
    "MODELS_DIR",
    "NEGATIVE_LABELS",
    "OUTPUTS_DIR",
    "POSITIVE_LABELS",
    "PersonDetection",
    "PROJECT_ROOT",
    "TEST_IMAGES_DIR",
    "WORK_CLOTHES_DIR",
    "WorkClothesConfig",
    "WorkClothesDetector",
    "annotate_prediction",
    "build_csi_gstreamer_pipeline",
    "iter_image_files",
    "open_camera",
    "save_prediction_visualization",
    "summarize_predictions",
]


def initialize_runtime() -> None:
    global _RUNTIME_INITIALIZED
    if _RUNTIME_INITIALIZED:
        return
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
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


def _normalize_label(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")


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


class WorkClothesDetector:
    def __init__(
        self,
        ppe_model_name: str = DEFAULT_PPE_MODEL_NAME,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
    ) -> None:
        initialize_runtime()

        self.feature_config = FEATURE_CONFIG
        self.ppe_model_path = Path(ppe_model_name)
        if not self.ppe_model_path.exists():
            raise FileNotFoundError(f"PPE 权重不存在: {self.ppe_model_path}")

        self.conf = conf
        self.iou = iou
        self.positive_labels = {_normalize_label(label) for label in POSITIVE_LABELS}
        self.negative_labels = {_normalize_label(label) for label in NEGATIVE_LABELS}
        self.model = YOLO(str(self.ppe_model_path))

    def _build_detection(self, class_name: str, confidence: float, box: Sequence[float], image_shape: tuple[int, int]) -> PersonDetection:
        height, width = image_shape
        clipped_box = _clip_box(box, width=width, height=height)
        normalized_class_name = _normalize_label(class_name)
        is_positive = normalized_class_name in self.positive_labels
        is_negative = normalized_class_name in self.negative_labels

        if is_positive:
            status = "with_safety_vest"
        elif is_negative:
            status = "without_safety_vest"
        else:
            status = "other_ppe"

        return PersonDetection(
            label="person",
            raw_class_name=class_name,
            confidence=round(float(confidence), 4),
            xyxy=[round(float(value), 2) for value in clipped_box],
            is_work_clothes=is_positive,
            status=status,
        )

    def predict_image(self, image_path: str | Path) -> ImagePrediction:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图片不存在: {image_path}")

        image = _load_image(image_path)
        image_shape = image.shape[:2]
        result = self.model.predict(
            source=str(image_path),
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )[0]

        detections: list[PersonDetection] = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            for box, cls, confidence in zip(boxes, classes, confidences, strict=True):
                class_name = str(self.model.names[int(cls)])
                detections.append(
                    self._build_detection(
                        class_name=class_name,
                        confidence=confidence,
                        box=box,
                        image_shape=image_shape,
                    )
                )

        return ImagePrediction(image_path=image_path, detections=detections)

    def predict_frame(
        self,
        frame: Any,
        conf: float | None = None,
        iou: float | None = None,
        verbose: bool = False,
        frame_name: str = "camera_frame",
    ) -> tuple[ImagePrediction, Any]:
        results = self.model.predict(
            source=frame,
            conf=self.conf if conf is None else conf,
            iou=self.iou if iou is None else iou,
            verbose=verbose,
        )
        result = results[0]
        detections: list[PersonDetection] = []
        if result.boxes is not None:
            image_shape = frame.shape[:2]
            boxes = result.boxes.xyxy.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            for box, cls, confidence in zip(boxes, classes, confidences, strict=True):
                class_name = str(self.model.names[int(cls)])
                detections.append(
                    self._build_detection(
                        class_name=class_name,
                        confidence=confidence,
                        box=box,
                        image_shape=image_shape,
                    )
                )
        return ImagePrediction(image_path=Path(frame_name), detections=detections), result

    def predict_directory(self, image_dir: str | Path) -> list[ImagePrediction]:
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"图片目录不存在: {image_dir}")
        return [self.predict_image(image_path) for image_path in iter_image_files(image_dir)]

    def run_camera_inference(
        self,
        camera: str | int = "csi",
        conf: float | None = None,
        iou: float | None = None,
        frame_log_interval: int = 10,
        window_name: str = "Safety Vest Detection",
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
                        raise RuntimeError(
                            f"连续 {failed_reads} 次读取摄像头帧失败: camera={camera}。"
                        )
                    continue
                failed_reads = 0

                frame_count += 1
                prediction, result = self.predict_frame(
                    frame,
                    conf=conf,
                    iou=iou,
                    verbose=False,
                    frame_name=f"camera:{camera}",
                )

                if frame_log_interval > 0 and frame_count % frame_log_interval == 0:
                    if prediction.detections:
                        for detection in prediction.detections:
                            print(
                                f"[DETECT] {detection.status} "
                                f"{detection.raw_class_name}: {detection.confidence:.2f}"
                            )
                    else:
                        print("[DETECT] No PPE detections in this frame.")

                annotated_frame = result.plot()
                cv2.imshow(window_name, annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord(quit_key.lower()):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


def _annotate_image(image: Any, prediction: ImagePrediction):
    if cv2 is None:
        return image

    for detection in prediction.detections:
        if detection.status == "with_safety_vest":
            color = (0, 180, 0)
        elif detection.status == "without_safety_vest":
            color = (0, 0, 255)
        else:
            color = (255, 165, 0)

        x1, y1, x2, y2 = [int(round(value)) for value in detection.xyxy]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label_text = f"{detection.raw_class_name} {detection.confidence:.2f}"
        cv2.putText(
            image,
            label_text,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
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


def iter_image_files(image_dir: str | Path) -> Iterable[Path]:
    image_dir = Path(image_dir)
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def summarize_predictions(predictions: Sequence[ImagePrediction]) -> dict[str, int | float]:
    total_images = len(predictions)
    detected_images = sum(1 for prediction in predictions if prediction.detected)
    total_detections = sum(len(prediction.detections) for prediction in predictions)
    positive_detections = sum(
        1
        for prediction in predictions
        for detection in prediction.detections
        if detection.status == "with_safety_vest"
    )
    negative_detections = sum(
        1
        for prediction in predictions
        for detection in prediction.detections
        if detection.status == "without_safety_vest"
    )
    return {
        "total_images": total_images,
        "detected_images": detected_images,
        "undetected_images": total_images - detected_images,
        "total_detections": total_detections,
        "positive_detections": positive_detections,
        "negative_detections": negative_detections,
        "avg_detections_per_image": round(total_detections / total_images, 4) if total_images else 0.0,
    }

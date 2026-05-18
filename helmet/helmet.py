from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Sequence

HELMET_DIR = Path(__file__).resolve().parent
TEST_IMAGES_DIR = HELMET_DIR / "imgs"
OUTPUTS_DIR = HELMET_DIR / "outputs"
MODELS_DIR = HELMET_DIR
YOLO_CONFIG_DIR = HELMET_DIR / ".ultralytics"

os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local environment
    cv2 = None

import torch
from ultralytics import YOLO

FEATURE_NAME = "helmet"
DISPLAY_NAME = "安全帽检测模型"
EXPECTED_LABELS = ("Hardhat", "NO-Hardhat")
DEFAULT_HF_MODEL_NAME = "keremberke/yolov8n-hard-hat-detection"
DEFAULT_MODEL_PATH = HELMET_DIR / "best.pt"
DEFAULT_MODEL_NAME = str(DEFAULT_MODEL_PATH)
DEFAULT_IMAGE_PATH = TEST_IMAGES_DIR / "helmet.jpg"
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_RUNTIME_INITIALIZED = False
_ORIGINAL_TORCH_LOAD = torch.load


@dataclass(frozen=True)
class HelmetConfig:
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
class ImagePrediction:
    image_path: Path
    detections: list[Detection]

    @property
    def detected(self) -> bool:
        return bool(self.detections)


FEATURE_CONFIG = HelmetConfig()

__all__ = [
    "DEFAULT_CONF",
    "DEFAULT_HF_MODEL_NAME",
    "DEFAULT_IMAGE_PATH",
    "DEFAULT_IOU",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_MODEL_PATH",
    "Detection",
    "DISPLAY_NAME",
    "EXPECTED_LABELS",
    "FEATURE_CONFIG",
    "FEATURE_NAME",
    "HELMET_DIR",
    "HelmetConfig",
    "HelmetDetector",
    "IMAGE_SUFFIXES",
    "ImagePrediction",
    "MODELS_DIR",
    "OUTPUTS_DIR",
    "TEST_IMAGES_DIR",
    "build_csi_gstreamer_pipeline",
    "filter_detections",
    "iter_image_files",
    "open_camera",
    "summarize_predictions",
]


def _torch_load_compat(*args, **kwargs):
    # Keep compatibility with checkpoints saved before PyTorch 2.6.
    kwargs.setdefault("weights_only", False)
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


def initialize_runtime() -> None:
    global _RUNTIME_INITIALIZED
    if _RUNTIME_INITIALIZED:
        return

    torch.load = _torch_load_compat
    YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _RUNTIME_INITIALIZED = True


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("未安装 opencv-python，无法使用摄像头实时识别。")


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


class HelmetDetector:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        conf: float | None = None,
        iou: float | None = None,
        agnostic_nms: bool = False,
        max_det: int = 1000,
    ) -> None:
        initialize_runtime()

        self.feature_config = FEATURE_CONFIG
        self.model_path = Path(model_name)
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        self.model = YOLO(str(self.model_path))
        self.model.overrides["conf"] = DEFAULT_CONF if conf is None else conf
        self.model.overrides["iou"] = DEFAULT_IOU if iou is None else iou
        self.model.overrides["agnostic_nms"] = agnostic_nms
        self.model.overrides["max_det"] = max_det

    @property
    def class_names(self) -> dict[int, str]:
        return self.model.model.names

    def _build_prediction(self, result: Any, image_path: Path) -> ImagePrediction:
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
        return self._build_prediction(results[0], image_path=image_path)

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
            conf=self.model.overrides["conf"] if conf is None else conf,
            iou=self.model.overrides["iou"] if iou is None else iou,
            verbose=verbose,
        )
        result = results[0]
        return self._build_prediction(result, image_path=Path(frame_name)), result

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

    def run_camera_inference(
        self,
        camera: str | int = "csi",
        conf: float | None = None,
        iou: float | None = None,
        frame_log_interval: int = 10,
        window_name: str = "Helmet Detection",
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
                                f"[DETECT] {detection.label}: {detection.confidence:.2f}"
                            )
                    else:
                        print("[DETECT] No objects in this frame.")

                annotated_frame = result.plot()
                cv2.imshow(window_name, annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord(quit_key.lower()):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


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

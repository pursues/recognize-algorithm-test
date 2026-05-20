from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

FALL_DOWN_DIR = Path(__file__).resolve().parent
TEST_IMAGES_DIR = FALL_DOWN_DIR / "imgs"
OUTPUTS_DIR = FALL_DOWN_DIR / "outputs"
MODELS_DIR = FALL_DOWN_DIR
YOLO_CONFIG_DIR = FALL_DOWN_DIR / ".ultralytics"

os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

try:
    import cv2
except ImportError:
    cv2 = None

import torch
from ultralytics import YOLO

FEATURE_NAME = "fall_down"
DISPLAY_NAME = "摔倒识别模型"
EXPECTED_LABELS = ("person", "fall")
DEFAULT_MODEL_NAME = "yolov8n.pt"  # 默认使用 yolov8n.pt
DEFAULT_CONF = 0.5
DEFAULT_IOU = 0.45
DEFAULT_FALL_THRESHOLD = 1.0 # 宽高比阈值
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_RUNTIME_INITIALIZED = False
_ORIGINAL_TORCH_LOAD = torch.load


@dataclass(frozen=True)
class FallDownConfig:
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
    is_fall: bool = False


@dataclass
class ImagePrediction:
    image_path: Path
    detections: list[Detection]

    @property
    def detected(self) -> bool:
        return bool(self.detections)
    
    @property
    def fall_detected(self) -> bool:
        return any(d.is_fall for d in self.detections)


FEATURE_CONFIG = FallDownConfig()

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


class FallDownDetector:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        conf: float | None = None,
        iou: float | None = None,
        fall_threshold: float = DEFAULT_FALL_THRESHOLD,
    ) -> None:
        initialize_runtime()

        self.feature_config = FEATURE_CONFIG
        self.model_path = model_name

        self.model = YOLO(str(self.model_path))
        self.model.overrides["conf"] = DEFAULT_CONF if conf is None else conf
        self.model.overrides["iou"] = DEFAULT_IOU if iou is None else iou
        self.fall_threshold = fall_threshold

    @property
    def class_names(self) -> dict[int, str]:
        return self.model.model.names

    def _is_fall(self, xyxy: list[float], keypoints: Any = None) -> bool:
        # 如果有关键点（yolov8n-pose.pt），可以使用更高级的逻辑
        # 这里使用通用的 bounding box 长宽比逻辑（适用于 yolov8n.pt）
        x1, y1, x2, y2 = xyxy
        width = x2 - x1
        height = y2 - y1
        if height == 0:
            return False
        aspect_ratio = width / height
        return aspect_ratio > self.fall_threshold

    def _build_prediction(self, result: Any, image_path: Path) -> ImagePrediction:
        boxes = result.boxes
        keypoints = result.keypoints if hasattr(result, 'keypoints') else None
        detections: list[Detection] = []

        if boxes is not None:
            for index in range(len(boxes)):
                cls_id = int(boxes.cls[index].item())
                label = self.class_names.get(cls_id, str(cls_id))
                
                # 只处理 'person' 类别，如果模型是其他特定的摔倒模型则不过滤
                if label != 'person' and label not in FEATURE_CONFIG.expected_labels:
                    continue
                    
                confidence = float(boxes.conf[index].item())
                xyxy = boxes.xyxy[index].tolist()
                
                # 获取该检测框的关键点（如果存在）
                kpts = keypoints[index] if keypoints is not None else None
                
                is_fall = self._is_fall(xyxy, kpts)
                
                # 如果检测到摔倒，修改标签为 fall
                display_label = "fall" if is_fall else label
                
                detections.append(
                    Detection(
                        label=display_label,
                        confidence=confidence,
                        xyxy=xyxy,
                        is_fall=is_fall
                    )
                )

        return ImagePrediction(image_path=image_path, detections=detections)

    def predict_image(
        self,
        image_path: str | Path,
        save: bool = False,
        project: str | Path | None = None,
        name: str = "predict",
    ) -> tuple[ImagePrediction, Any]:
        image_path = Path(image_path)
        predict_kwargs = {
            "source": str(image_path),
            "save": False, # 我们自己处理渲染
            "name": name,
            "exist_ok": True,
        }
        if project is not None:
            predict_kwargs["project"] = str(project)

        results = self.model.predict(**predict_kwargs)
        prediction = self._build_prediction(results[0], image_path=image_path)
        return prediction, results[0]

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

    def _plot_custom(self, result: Any, prediction: ImagePrediction, frame: Any) -> Any:
        import copy
        annotated_frame = copy.deepcopy(frame)
        for det in prediction.detections:
            x1, y1, x2, y2 = map(int, det.xyxy)
            color = (0, 0, 255) if det.is_fall else (0, 255, 0) # 摔倒红色，正常绿色
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label_text = f"{det.label} {det.confidence:.2f}"
            cv2.putText(annotated_frame, label_text, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return annotated_frame

    def run_camera_inference(
        self,
        camera: str | int = "csi",
        conf: float | None = None,
        iou: float | None = None,
        frame_log_interval: int = 10,
        window_name: str = "Fall Down Detection",
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
                    if prediction.fall_detected:
                        print(f"[DETECT] Fall Down Detected!")
                    elif prediction.detections:
                        print(f"[DETECT] Normal person detected.")

                annotated_frame = self._plot_custom(result, prediction, frame)
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

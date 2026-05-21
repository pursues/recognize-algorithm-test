from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Sequence

NOT_MASK_DIR = Path(__file__).resolve().parent
TEST_IMAGES_DIR = NOT_MASK_DIR / "imgs"
OUTPUTS_DIR = NOT_MASK_DIR / "outputs"
MODELS_DIR = NOT_MASK_DIR / "models"
DEFAULT_LOCAL_MODEL_PATH = MODELS_DIR / "face-mask-detection"

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local environment
    cv2 = None

from PIL import Image
import torch
from torchvision.models import swin_t
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.functional import normalize
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms.functional import resize

FEATURE_NAME = "not_mask"
DISPLAY_NAME = "未戴口罩识别"
EXPECTED_LABELS = (
    "WithoutMask",
    "WithMask",
    "Face_Mask Not_Found",
    "Face_Mask Found",
)
DEFAULT_MODEL_PATH = DEFAULT_LOCAL_MODEL_PATH
DEFAULT_MODEL_NAME = str(DEFAULT_MODEL_PATH)
DEFAULT_IMAGE_PATH = TEST_IMAGES_DIR / "not-mask-leijun.jpg"
DEFAULT_SCORE_THRESHOLD = 0.6
DEFAULT_FACE_PADDING_RATIO = 0.18
DEFAULT_MIN_FACE_SIZE = 40
DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 720
DEFAULT_CAMERA_FRAMERATE = 30
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_RUNTIME_INITIALIZED = False


@dataclass(frozen=True)
class NotMaskConfig:
    name: str = FEATURE_NAME
    display_name: str = DISPLAY_NAME
    expected_labels: tuple[str, ...] = EXPECTED_LABELS
    default_score_threshold: float = DEFAULT_SCORE_THRESHOLD


@dataclass
class FaceDetection:
    label: str
    confidence: float
    xyxy: list[float]
    is_no_mask: bool
    source: str
    label_scores: dict[str, float]


@dataclass
class ImagePrediction:
    image_path: Path
    face_detections: list[FaceDetection]

    @property
    def detected(self) -> bool:
        return any(detection.is_no_mask for detection in self.face_detections)


FEATURE_CONFIG = NotMaskConfig()

__all__ = [
    "DEFAULT_FACE_PADDING_RATIO",
    "DEFAULT_IMAGE_PATH",
    "DEFAULT_LOCAL_MODEL_PATH",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_MIN_FACE_SIZE",
    "DEFAULT_SCORE_THRESHOLD",
    "DISPLAY_NAME",
    "EXPECTED_LABELS",
    "FEATURE_CONFIG",
    "FEATURE_NAME",
    "FaceDetection",
    "IMAGE_SUFFIXES",
    "ImagePrediction",
    "MODELS_DIR",
    "NOT_MASK_DIR",
    "NotMaskConfig",
    "NotMaskDetector",
    "OUTPUTS_DIR",
    "TEST_IMAGES_DIR",
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

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    _RUNTIME_INITIALIZED = True


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("未安装 opencv-python，无法使用摄像头实时识别或结果可视化。")


def _load_image(image_path: Path):
    _require_cv2()
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"无法读取图片: {image_path}")
    return image


def _build_face_detector():
    if cv2 is None:
        return None
    yunet_path = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
    if not yunet_path.exists():
        # 回退到 Haar 级联作为备用方案
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists():
            return None
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            return None
        return cascade
    # 使用 YuNet 预训练模型，精度和速度均远超 Haar
    return cv2.FaceDetectorYN.create(str(yunet_path), "", (320, 320))


def build_csi_gstreamer_pipeline(
    width: int = DEFAULT_CAMERA_WIDTH,
    height: int = DEFAULT_CAMERA_HEIGHT,
    framerate: int = DEFAULT_CAMERA_FRAMERATE,
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
    width: int = DEFAULT_CAMERA_WIDTH,
    height: int = DEFAULT_CAMERA_HEIGHT,
    framerate: int = DEFAULT_CAMERA_FRAMERATE,
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


def _detect_faces(face_detector: Any, image: Any, min_face_size: int) -> list[list[float]]:
    if face_detector is None or cv2 is None:
        return []

    if hasattr(face_detector, "setInputSize"):
        # YuNet 检测
        height, width = image.shape[:2]
        face_detector.setInputSize((width, height))
        _, faces = face_detector.detect(image)
        if faces is None:
            return []
        detected_boxes = []
        for face in faces:
            # face: [x, y, w, h, ...]
            x, y, w, h = face[:4]
            score = face[-1]
            if w >= min_face_size and h >= min_face_size and score >= 0.6:
                detected_boxes.append([float(x), float(y), float(x + w), float(y + h)])
        return detected_boxes
    else:
        # Haar 级联检测 (备用)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_face_size, min_face_size),
        )
        return [[float(x), float(y), float(x + w), float(y + h)] for x, y, w, h in faces]


def _expand_box(box: Sequence[float], width: int, height: int, padding_ratio: float) -> list[int]:
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = box_width * padding_ratio
    pad_y = box_height * padding_ratio
    return [
        max(int(round(x1 - pad_x)), 0),
        max(int(round(y1 - pad_y)), 0),
        min(int(round(x2 + pad_x)), width),
        min(int(round(y2 + pad_y)), height),
    ]


def _normalize_label(label: str) -> str:
    return (
        label.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def _is_no_mask_label(label: str) -> bool:
    normalized = _normalize_label(label)
    return any(
        token in normalized
        for token in (
            "not_found",
            "no_mask",
            "without_mask",
            "withoutmask",
            "mask_not_found",
            "nomask",
        )
    )


def _label_score_mapping(id2label: dict[int, str], scores: Sequence[float]) -> dict[str, float]:
    return {
        id2label[index]: round(float(score), 4)
        for index, score in enumerate(scores)
        if index in id2label
    }


def _read_json_file(file_path: Path) -> dict[str, Any]:
    return json.loads(file_path.read_text(encoding="utf-8"))


def _load_local_model_metadata(model_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = model_dir / "config.json"
    preprocessor_path = model_dir / "preprocessor_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"模型配置文件不存在: {config_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"预处理配置文件不存在: {preprocessor_path}")
    return _read_json_file(config_path), _read_json_file(preprocessor_path)


def _map_hf_swin_state_dict_to_torchvision(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    stage_features = {0: 1, 1: 3, 2: 5, 3: 7}
    mapped: dict[str, torch.Tensor] = {}
    qkv_parts: dict[tuple[int, int, str], dict[str, torch.Tensor]] = {}

    for key, value in state_dict.items():
        if key.startswith("swin.embeddings.patch_embeddings.projection."):
            mapped[key.replace("swin.embeddings.patch_embeddings.projection", "features.0.0")] = value
            continue
        if key.startswith("swin.embeddings.norm."):
            mapped[key.replace("swin.embeddings.norm", "features.0.2")] = value
            continue
        if key.startswith("swin.layernorm."):
            mapped[key.replace("swin.layernorm", "norm")] = value
            continue
        if key.startswith("classifier."):
            mapped[key.replace("classifier", "head")] = value
            continue
        if not key.startswith("swin.encoder.layers."):
            continue

        rest = key[len("swin.encoder.layers.") :]
        stage_str, rest = rest.split(".", 1)
        stage = int(stage_str)
        base = f"features.{stage_features[stage]}"

        if rest.startswith("blocks."):
            _, block_str, rest = rest.split(".", 2)
            block = int(block_str)
            prefix = f"{base}.{block}"

            if rest.startswith("layernorm_before."):
                mapped[f"{prefix}.norm1.{rest.split('.', 1)[1]}"] = value
            elif rest.startswith("layernorm_after."):
                mapped[f"{prefix}.norm2.{rest.split('.', 1)[1]}"] = value
            elif rest == "attention.self.relative_position_bias_table":
                mapped[f"{prefix}.attn.relative_position_bias_table"] = value
            elif rest == "attention.self.relative_position_index":
                continue
            elif rest.startswith("attention.self.query."):
                qkv_parts.setdefault((stage, block, rest.rsplit(".", 1)[1]), {})["q"] = value
            elif rest.startswith("attention.self.key."):
                qkv_parts.setdefault((stage, block, rest.rsplit(".", 1)[1]), {})["k"] = value
            elif rest.startswith("attention.self.value."):
                qkv_parts.setdefault((stage, block, rest.rsplit(".", 1)[1]), {})["v"] = value
            elif rest.startswith("attention.output.dense."):
                mapped[f"{prefix}.attn.proj.{rest.rsplit('.', 1)[1]}"] = value
            elif rest.startswith("intermediate.dense."):
                mapped[f"{prefix}.mlp.0.{rest.split('.', 2)[2]}"] = value
            elif rest.startswith("output.dense."):
                mapped[f"{prefix}.mlp.3.{rest.split('.', 2)[2]}"] = value
            continue

        if rest.startswith("downsample."):
            down_prefix = f"features.{stage_features[stage] + 1}"
            if rest.startswith("downsample.reduction."):
                mapped[f"{down_prefix}.reduction.{rest.split('.', 2)[2]}"] = value
            elif rest.startswith("downsample.norm."):
                mapped[f"{down_prefix}.norm.{rest.split('.', 2)[2]}"] = value

    for (stage, block, suffix), parts in qkv_parts.items():
        if not {"q", "k", "v"} <= set(parts):
            raise RuntimeError(f"Swin 权重缺少完整的 QKV 参数: stage={stage}, block={block}, suffix={suffix}")
        mapped[f"features.{stage_features[stage]}.{block}.attn.qkv.{suffix}"] = torch.cat(
            [parts["q"], parts["k"], parts["v"]],
            dim=0,
        )

    return mapped


def _load_local_swin_model(model_dir: Path, num_classes: int) -> torch.nn.Module:
    model_file = model_dir / "pytorch_model.bin"
    if not model_file.exists():
        raise FileNotFoundError(f"模型权重文件不存在: {model_file}")

    raw_state_dict = torch.load(model_file, map_location="cpu")
    state_dict = raw_state_dict.get("state_dict", raw_state_dict)
    mapped_state_dict = _map_hf_swin_state_dict_to_torchvision(state_dict)
    model = swin_t(weights=None, num_classes=num_classes)
    missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=False)
    missing_keys = [
        key for key in missing_keys if not key.endswith("attn.relative_position_index")
    ]
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Swin 本地权重映射失败: "
            f"missing_keys={missing_keys[:10]}, unexpected_keys={unexpected_keys[:10]}"
        )
    return model


class NotMaskDetector:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        face_padding_ratio: float = DEFAULT_FACE_PADDING_RATIO,
        min_face_size: int = DEFAULT_MIN_FACE_SIZE,
    ) -> None:
        initialize_runtime()

        self.feature_config = FEATURE_CONFIG
        self.model_name = model_name
        self.score_threshold = score_threshold
        self.face_padding_ratio = face_padding_ratio
        self.min_face_size = min_face_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = Path(model_name)
        if not self.model_path.exists():
            raise FileNotFoundError(f"本地模型目录不存在: {self.model_path}")
        self.model_config, self.preprocessor_config = _load_local_model_metadata(self.model_path)
        config_labels = self.model_config.get("id2label", {}) or {}
        self.id2label = {int(index): label for index, label in config_labels.items()}
        self.image_size = self.preprocessor_config.get("size", {})
        self.image_height = int(self.image_size.get("height", self.model_config.get("image_size", 224)))
        self.image_width = int(self.image_size.get("width", self.model_config.get("image_size", 224)))
        self.image_mean = list(self.preprocessor_config.get("image_mean", [0.485, 0.456, 0.406]))
        self.image_std = list(self.preprocessor_config.get("image_std", [0.229, 0.224, 0.225]))
        self.model = _load_local_swin_model(
            self.model_path,
            num_classes=max(len(self.id2label), 1),
        )
        self.model.eval()
        self.model.to(self.device)
        self.face_detector = _build_face_detector()

    def _classify_pil_image(self, image: Image.Image) -> tuple[str, float, bool, dict[str, float]]:
        image = image.convert("RGB")
        image = resize(
            image,
            [self.image_height, self.image_width],
            interpolation=InterpolationMode.BICUBIC,
        )
        pixel_values = pil_to_tensor(image).float() / 255.0
        pixel_values = normalize(pixel_values, mean=self.image_mean, std=self.image_std)
        pixel_values = pixel_values.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(pixel_values)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()

        label_scores = _label_score_mapping(self.id2label, probabilities)
        if label_scores:
            label, confidence = max(label_scores.items(), key=lambda item: item[1])
        else:
            best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
            label = str(best_index)
            confidence = float(probabilities[best_index])

        is_no_mask = _is_no_mask_label(label) and confidence >= self.score_threshold
        return label, float(confidence), is_no_mask, label_scores

    def _classify_face_crop(
        self,
        image_rgb: Any,
        face_box: Sequence[float],
        source: str,
    ) -> FaceDetection:
        height, width = image_rgb.shape[:2]
        x1, y1, x2, y2 = _expand_box(
            face_box,
            width=width,
            height=height,
            padding_ratio=self.face_padding_ratio,
        )
        face_crop = image_rgb[y1:y2, x1:x2]
        pil_image = Image.fromarray(face_crop)
        label, confidence, is_no_mask, label_scores = self._classify_pil_image(pil_image)
        return FaceDetection(
            label=label,
            confidence=round(confidence, 4),
            xyxy=[round(float(value), 2) for value in (x1, y1, x2, y2)],
            is_no_mask=is_no_mask,
            source=source,
            label_scores=label_scores,
        )

    def predict_image(self, image_path: str | Path) -> ImagePrediction:
        image_path = Path(image_path)
        image_bgr = _load_image(image_path)
        return self._predict_from_bgr_image(image_bgr, image_path=image_path)

    def predict_directory(self, image_dir: str | Path) -> list[ImagePrediction]:
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"图片目录不存在: {image_dir}")
        return [self.predict_image(image_path) for image_path in iter_image_files(image_dir)]

    def _predict_from_bgr_image(self, image_bgr: Any, image_path: Path) -> ImagePrediction:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        face_boxes = _detect_faces(
            self.face_detector,
            image=image_bgr,
            min_face_size=self.min_face_size,
        )

        detections = [
            self._classify_face_crop(image_rgb, face_box, source="face")
            for face_box in face_boxes
        ]

        if not detections:
            full_frame_box = [0.0, 0.0, float(image_rgb.shape[1]), float(image_rgb.shape[0])]
            detections.append(
                self._classify_face_crop(
                    image_rgb,
                    face_box=full_frame_box,
                    source="full_image_fallback",
                )
            )

        return ImagePrediction(image_path=image_path, face_detections=detections)

    def predict_frame(
        self,
        frame: Any,
        frame_name: str = "camera_frame",
    ) -> ImagePrediction:
        return self._predict_from_bgr_image(frame, image_path=Path(frame_name))

    def run_camera_inference(
        self,
        camera: str | int = "csi",
        frame_log_interval: int = 10,
        window_name: str = "Not Mask Detection",
        width: int = DEFAULT_CAMERA_WIDTH,
        height: int = DEFAULT_CAMERA_HEIGHT,
        framerate: int = DEFAULT_CAMERA_FRAMERATE,
        flip_method: int = 0,
        quit_key: str = "q",
        warmup_frames: int = 5,
        max_failed_reads: int = 30,
        process_every_n_frames: int = 5,  # 新增跳帧参数：每隔几帧推理一次
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

        print(f"实时识别已启动，摄像头={camera}，每 {process_every_n_frames} 帧推理一次，按 '{quit_key.upper()}' 退出。")
        frame_count = 0
        failed_reads = 0
        last_prediction = None  # 缓存上一帧的推理结果
        last_annotated_frame = None

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
                
                # 核心跳帧逻辑：减少对 GPU/CPU 的瞬时过载
                if frame_count % process_every_n_frames == 1 or last_prediction is None:
                    last_prediction = self.predict_frame(
                        frame,
                        frame_name=f"camera:{camera}",
                    )
                    last_annotated_frame = _annotate_image(frame.copy(), last_prediction)
                else:
                    # 对于跳过的帧，把上次的框画在当前帧上
                    last_annotated_frame = _annotate_image(frame.copy(), last_prediction)

                if frame_log_interval > 0 and frame_count % frame_log_interval == 0:
                    if last_prediction.face_detections:
                        for detection in last_prediction.face_detections:
                            status = "NO_MASK" if detection.is_no_mask else "MASK"
                            print(
                                f"[DETECT] {status} {detection.label}: {detection.confidence:.2f}"
                            )
                    else:
                        print("[DETECT] No faces in this frame.")

                cv2.imshow(window_name, last_annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord(quit_key.lower()):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


def _annotate_image(image: Any, prediction: ImagePrediction):
    if cv2 is None:
        return image

    for detection in prediction.face_detections:
        color = (0, 0, 255) if detection.is_no_mask else (0, 180, 0)
        x1, y1, x2, y2 = [int(round(value)) for value in detection.xyxy]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label_text = (
            f"{detection.label} {detection.confidence:.2f}"
            if detection.source == "face"
            else f"{detection.label} {detection.confidence:.2f} ({detection.source})"
        )
        cv2.putText(
            image,
            label_text,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
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
    total_faces = sum(len(prediction.face_detections) for prediction in predictions)
    total_no_mask_faces = sum(
        1
        for prediction in predictions
        for detection in prediction.face_detections
        if detection.is_no_mask
    )
    return {
        "total_images": total_images,
        "detected_images": detected_images,
        "undetected_images": total_images - detected_images,
        "total_faces": total_faces,
        "no_mask_faces": total_no_mask_faces,
        "avg_faces_per_image": round(total_faces / total_images, 4) if total_images else 0.0,
    }

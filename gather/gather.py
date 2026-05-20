from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

GATHER_DIR = Path(__file__).resolve().parent
TEST_IMAGES_DIR = GATHER_DIR / "imgs"
OUTPUTS_DIR = GATHER_DIR / "outputs"
YOLO_CONFIG_DIR = GATHER_DIR / ".ultralytics"

os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

import torch
from ultralytics import YOLO

FEATURE_NAME = "gather"
DISPLAY_NAME = "人员聚集识别模型"
EXPECTED_LABELS = ("person",)
DEFAULT_MODEL_NAME = "yolov8n.pt"
DEFAULT_MODEL_PATH = GATHER_DIR / DEFAULT_MODEL_NAME
DEFAULT_CONF = 0.35
DEFAULT_IOU = 0.45
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_RUNTIME_INITIALIZED = False
_ORIGINAL_TORCH_LOAD = torch.load


@dataclass(frozen=True)
class GatherConfig:
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
    track_id: int | None = None


@dataclass
class GatherGroup:
    group_id: int
    detections: list[Detection]


@dataclass
class ImagePrediction:
    image_path: Path
    detections: list[Detection]
    gather_groups: list[GatherGroup]

    @property
    def detected(self) -> bool:
        return bool(self.detections)

    @property
    def has_gathering(self) -> bool:
        return bool(self.gather_groups)


FEATURE_CONFIG = GatherConfig()

__all__ = [
    "DEFAULT_CONF",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_IOU",
    "Detection",
    "DISPLAY_NAME",
    "EXPECTED_LABELS",
    "FEATURE_CONFIG",
    "FEATURE_NAME",
    "GATHER_DIR",
    "GatherConfig",
    "GatherDetector",
    "GatherGroup",
    "IMAGE_SUFFIXES",
    "ImagePrediction",
    "OUTPUTS_DIR",
    "TEST_IMAGES_DIR",
    "build_csi_gstreamer_pipeline",
    "open_camera",
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


def calculate_distance(box1: list[float], box2: list[float]) -> float:
    """计算两个边界框中心点之间的距离"""
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2
    return math.sqrt((x1_center - x2_center) ** 2 + (y1_center - y2_center) ** 2)


def find_gather_groups(
    detections: list[Detection], distance_threshold: float, min_gather_count: int
) -> list[GatherGroup]:
    """基于距离对检测到的人员进行聚类"""
    if not detections:
        return []

    # 简单并查集或连通图组件来寻找群体
    n = len(detections)
    parent = list(range(n))

    def find(i: int) -> int:
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int) -> None:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # 如果两个人之间的距离小于阈值，认为他们在一起
    for i in range(n):
        for j in range(i + 1, n):
            if calculate_distance(detections[i].xyxy, detections[j].xyxy) < distance_threshold:
                union(i, j)

    # 收集每一组的人员
    groups: dict[int, list[Detection]] = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(detections[i])

    # 过滤出人数大于等于最小聚集人数的组
    gather_groups = []
    group_id_counter = 1
    for root, members in groups.items():
        if len(members) >= min_gather_count:
            gather_groups.append(GatherGroup(group_id=group_id_counter, detections=members))
            group_id_counter += 1

    return gather_groups


class GatherDetector:
    def __init__(
        self,
        model_name: str | Path = DEFAULT_MODEL_PATH,
        conf: float | None = None,
        iou: float | None = None,
        distance_threshold: float = 150.0,
        min_gather_count: int = 3,
    ) -> None:
        initialize_runtime()

        self.feature_config = FEATURE_CONFIG
        self.model_path = Path(model_name)
        
        # 如果本地没有模型文件，且是默认名称，则通过ultralytics自动下载到当前目录或使用全局缓存
        # 这里使用 yolov8n.pt，YOLO 会自动处理下载
        self.model = YOLO(str(self.model_path))
        self.model.overrides["conf"] = DEFAULT_CONF if conf is None else conf
        self.model.overrides["iou"] = DEFAULT_IOU if iou is None else iou
        self.model.overrides["classes"] = [0]  # COCO class 0 is person

        self.distance_threshold = distance_threshold
        self.min_gather_count = min_gather_count

    @property
    def class_names(self) -> dict[int, str]:
        return self.model.model.names

    def _build_prediction(self, result: Any, image_path: Path) -> ImagePrediction:
        boxes = result.boxes
        detections: list[Detection] = []

        if boxes is not None:
            for index in range(len(boxes)):
                cls_id = int(boxes.cls[index].item())
                label = self.class_names.get(cls_id, str(cls_id))
                # 再次确认只处理 person
                if label not in FEATURE_CONFIG.expected_labels:
                    continue

                confidence = float(boxes.conf[index].item())
                xyxy = boxes.xyxy[index].tolist()
                track_id = int(boxes.id[index].item()) if boxes.id is not None else None
                
                detections.append(
                    Detection(
                        label=label,
                        confidence=confidence,
                        xyxy=xyxy,
                        track_id=track_id,
                    )
                )

        gather_groups = find_gather_groups(
            detections, self.distance_threshold, self.min_gather_count
        )

        return ImagePrediction(
            image_path=image_path,
            detections=detections,
            gather_groups=gather_groups,
        )

    def draw_gathering(self, frame: Any, prediction: ImagePrediction) -> Any:
        """在图像上绘制聚集情况"""
        annotated_frame = frame.copy()
        
        # 绘制所有人的框 (绿色)
        for det in prediction.detections:
            x1, y1, x2, y2 = map(int, det.xyxy)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"person {det.confidence:.2f}"
            if det.track_id is not None:
                label_text += f" id:{det.track_id}"
            cv2.putText(annotated_frame, label_text, (x1, max(y1 - 10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 绘制聚集群组 (红色框，以及组间连线或外接框)
        for group in prediction.gather_groups:
            # 找到聚集组的外接矩形
            x_mins = [det.xyxy[0] for det in group.detections]
            y_mins = [det.xyxy[1] for det in group.detections]
            x_maxs = [det.xyxy[2] for det in group.detections]
            y_maxs = [det.xyxy[3] for det in group.detections]
            
            gx1, gy1 = int(min(x_mins)), int(min(y_mins))
            gx2, gy2 = int(max(x_maxs)), int(max(y_maxs))
            
            # 画一个大红框包围聚集的人
            cv2.rectangle(annotated_frame, (gx1, gy1), (gx2, gy2), (0, 0, 255), 3)
            
            # 标注聚集警告
            warn_text = f"WARNING: Gathering! Group {group.group_id} ({len(group.detections)} people)"
            cv2.putText(annotated_frame, warn_text, (gx1, max(gy1 - 15, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 把聚集的人框标红
            for det in group.detections:
                x1, y1, x2, y2 = map(int, det.xyxy)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return annotated_frame

    def predict_image(
        self,
        image_path: str | Path,
        save: bool = False,
        output_dir: str | Path | None = None,
        use_tracking: bool = False,
    ) -> tuple[ImagePrediction, Any]:
        image_path = Path(image_path)
        _require_cv2()
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"无法读取图片: {image_path}")

        # 使用 ByteTrack 进行跟踪，或者仅仅使用 predict 进行检测
        results = None
        if use_tracking:
            try:
                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )
            except (ImportError, ModuleNotFoundError) as e:
                if "lap" in str(e).lower() or "lap" in repr(e).lower():
                    print("警告: 缺少 'lap' 库，无法使用目标跟踪功能。已自动降级为单帧检测。")
                    use_tracking = False
                else:
                    raise

        if not use_tracking or results is None:
            results = self.model.predict(
                source=frame,
                verbose=False,
            )
            
        prediction = self._build_prediction(results[0], image_path=image_path)
        
        annotated_frame = self.draw_gathering(frame, prediction)
        
        if save and output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / image_path.name
            cv2.imwrite(str(save_path), annotated_frame)
            
        return prediction, annotated_frame

    def run_camera_inference(
        self,
        camera: str | int = "csi",
        conf: float | None = None,
        iou: float | None = None,
        frame_log_interval: int = 10,
        window_name: str = "Gather Detection",
        width: int = 1280,
        height: int = 720,
        framerate: int = 30,
        flip_method: int = 0,
        quit_key: str = "q",
        warmup_frames: int = 5,
        max_failed_reads: int = 30,
        use_tracking: bool = False,
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

        print(f"实时聚集识别已启动 (使用{'ByteTrack跟踪' if use_tracking else '单帧检测'})，摄像头={camera}，按 '{quit_key.upper()}' 退出。")
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
                
                results = None
                if use_tracking:
                    try:
                        results = self.model.track(
                            source=frame,
                            persist=True,
                            tracker="bytetrack.yaml",
                            conf=self.model.overrides["conf"] if conf is None else conf,
                            iou=self.model.overrides["iou"] if iou is None else iou,
                            verbose=False,
                        )
                    except (ImportError, ModuleNotFoundError) as e:
                        if "lap" in str(e).lower() or "lap" in repr(e).lower():
                            if frame_count == 1:
                                print("警告: 缺少 'lap' 库，无法使用目标跟踪功能。已自动降级为单帧检测。")
                            use_tracking = False
                        else:
                            raise

                if not use_tracking or results is None:
                    results = self.model.predict(
                        source=frame,
                        conf=self.model.overrides["conf"] if conf is None else conf,
                        iou=self.model.overrides["iou"] if iou is None else iou,
                        verbose=False,
                    )
                    
                prediction = self._build_prediction(results[0], image_path=Path(f"camera:{camera}"))
                annotated_frame = self.draw_gathering(frame, prediction)

                if frame_log_interval > 0 and frame_count % frame_log_interval == 0:
                    if prediction.has_gathering:
                        print(f"[DETECT] {len(prediction.gather_groups)} gatherings detected! Total people gathered: {sum(len(g.detections) for g in prediction.gather_groups)}")
                    else:
                        print(f"[DETECT] Normal. {len(prediction.detections)} people detected.")

                cv2.imshow(window_name, annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord(quit_key.lower()):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

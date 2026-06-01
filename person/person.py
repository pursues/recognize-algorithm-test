from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any
import threading
import datetime

PERSON_DIR = Path(__file__).resolve().parent
LIBS_DIR = PERSON_DIR / "libs"
TEST_IMAGES_DIR = PERSON_DIR / "imgs"
OUTPUTS_DIR = PERSON_DIR / "outputs"
YOLO_CONFIG_DIR = PERSON_DIR / ".ultralytics"

if LIBS_DIR.exists():
    sys.path.insert(0, str(LIBS_DIR))

os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

try:
    import cv2
except ImportError:
    cv2 = None

import torch
from ultralytics import YOLO

FEATURE_NAME = "person"
DISPLAY_NAME = "人员识别模型"
EXPECTED_LABELS = ("person",)
DEFAULT_MODEL_NAME = "yolov8n.pt"
DEFAULT_MODEL_PATH = PERSON_DIR / DEFAULT_MODEL_NAME
DEFAULT_CONF = 0.35
DEFAULT_IOU = 0.45
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_RUNTIME_INITIALIZED = False
_ORIGINAL_TORCH_LOAD = torch.load


@dataclass(frozen=True)
class PersonConfig:
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
class ImagePrediction:
    image_path: Path
    detections: list[Detection]

    @property
    def detected(self) -> bool:
        return bool(self.detections)


FEATURE_CONFIG = PersonConfig()

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
    "PERSON_DIR",
    "PersonConfig",
    "ImagePrediction",
    "OUTPUTS_DIR",
    "TEST_IMAGES_DIR",
    "build_csi_gstreamer_pipeline",
    "build_hk_gstreamer_pipeline",
    "open_camera",
    "send_alert",
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


def build_hk_gstreamer_pipeline(
    rtsp_url: str = "rtsp://admin:JIANGhd99110@192.168.5.113:554/Streaming/Channels/101",
    latency: int = 200,
) -> str:
    return (
        f"rtspsrc location={rtsp_url} latency={latency} protocols=tcp ! "
        "rtph265depay ! h265parse ! "
        "nvv4l2decoder ! "
        "nvvidconv ! "
        "video/x-raw,format=BGRx ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=1 max-buffers=2"
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
    if camera == "hk":
        rtsp_url = os.environ.get(
            "HK_RTSP_URL",
            "rtsp://admin:JIANGhd99110@192.168.5.113:554/Streaming/Channels/101"
        )
        gst_str = build_hk_gstreamer_pipeline(rtsp_url=rtsp_url)
        cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        for _ in range(5):
            if cap.isOpened():
                break
            cap.release()
            import time
            time.sleep(2)
            cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        return cap
    if camera == "usb":
        return cv2.VideoCapture(0, cv2.CAP_ANY)
    if isinstance(camera, str) and camera.isdigit():
        return cv2.VideoCapture(int(camera))
    raise ValueError(f"不支持的摄像头类型: {camera}，可选值为 csi、usb、hk 或数字序号")


class AlertClient:
    def __init__(self):
        self.endpoint = 'https://oss-rg-china-mainland.aliyuncs.com'
        self.bucket_name = 'huitianfile'
        self.access_key_id = os.environ.get('ALIYUN_ACCESS_KEY_ID', '')
        self.access_key_secret = os.environ.get('ALIYUN_ACCESS_KEY_SECRET', '')
        
        import oss2
        auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        self.bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)
        
        self.alarm_counter = 0
        self.counter_lock = threading.Lock()
        self.last_date = datetime.datetime.now().strftime("%Y%m%d")
        
        self.alert_api_url = "http://192.168.5.53:7250/api/openapi/behavior/alarms"
        
    def _generate_alarm_no(self) -> str:
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        timestamp_ms = int(datetime.datetime.now().timestamp() * 1000)
        return f"ALARM-{current_date}-{timestamp_ms}"
    
    def upload_to_oss(self, image_path: str | Path, object_name: str | None = None) -> str:
        import oss2
        image_path = Path(image_path)
        
        if object_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            object_name = f"person_alert/{timestamp}_{image_path.name}"
        
        result = self.bucket.put_object_from_file(object_name, str(image_path))
        
        if result.status == 200:
            oss_url = f"https://{self.bucket_name}.{self.endpoint.replace('https://', '')}/{object_name}"
            print(f"[OSS] 上传成功: {oss_url}")
            return oss_url
        else:
            raise RuntimeError(f"OSS 上传失败，状态码: {result.status}")
    
    def send_alert_request(self, alarm_data: dict) -> dict:
        import requests
        try:
            headers = {
                "Content-Type": "application/json",
                "appKey": "PdvudIo0fEacM8Pls88U9fsDjMwhqQMh"
            }
            response = requests.post(
                self.alert_api_url,
                json=alarm_data,
                headers=headers,
                timeout=10000
            )
            response.raise_for_status()
            result = response.json()
            print(f"[API] 警报接口调用成功: {result}")
            return result
        except Exception as e:
            print(f"[API] 警报接口调用失败: {e}")
            return {"status": "error", "message": str(e)}


_alert_client = None
_alert_client_lock = threading.Lock()


def get_alert_client() -> AlertClient:
    global _alert_client
    if _alert_client is None:
        with _alert_client_lock:
            if _alert_client is None:
                _alert_client = AlertClient()
    return _alert_client


def send_alert(detection_count: int, image_path: str = "", frame_info: dict | None = None, frame: Any = None) -> dict:
    """
    真实警报接口 - 当检测到人员时调用
    
    参数:
        detection_count: 检测到的人员数量
        image_path: 触发警报的图片路径或来源
        frame_info: 额外的帧信息（用于实时识别时传递）
        frame: 当前帧图像数据（用于实时识别时保存到本地再上传）
    
    返回:
        警报接口响应结果
    """
    try:
        client = get_alert_client()
        
        alarm_no = client._generate_alarm_no()
        alarm_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        camera_location = "A区北门"
        if frame_info and "camera" in frame_info:
            camera_location = f"{frame_info['camera']}摄像头"
        
        oss_url = ""
        upload_image_path = image_path
        
        if frame is not None and frame_info:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            camera_name = frame_info.get("camera", "unknown")
            temp_filename = f"person_{timestamp}_{camera_name}.jpg"
            temp_path = OUTPUTS_DIR / temp_filename
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(temp_path), frame)
            upload_image_path = str(temp_path)
            print(f"[ALERT] 已保存临时图片: {temp_path}")
        
        if upload_image_path and Path(upload_image_path).exists():
            try:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                object_name = f"person_alert/{timestamp}_{Path(upload_image_path).name}"
                oss_url = client.upload_to_oss(upload_image_path, object_name)
            except Exception as e:
                print(f"[OSS] 上传失败: {e}")
                oss_url = ""
            finally:
                if frame is not None and frame_info:
                    try:
                        Path(upload_image_path).unlink()
                        print(f"[ALERT] 已清理临时图片: {upload_image_path}")
                    except Exception as e:
                        print(f"[ALERT] 清理临时图片失败: {e}")
        
        raw_data = f"confidence\":\"0.93\",\"cameraPoint\":\"{camera_location}"
        
        alarm_data = {
            "alarmNo": alarm_no,
            "sourceSystem": "IOT_PLATFORM",
            "deviceCode": "BEHAVIOR_CAMERA_001",
            "abnormalType": "ILLEGAL_INTRUSION",
            "alarmLevel": "L3",
            "currentValue": "-",
            "evidenceUrl": oss_url,
            "alarmTime": alarm_time,
            "rawData": raw_data
        }
        
        print(f"[ALERT] 警报触发: 检测到 {detection_count} 个人员")
        print(f"[ALERT] 警报编号: {alarm_no}")
        print(f"[ALERT] 证据URL: {oss_url}")
        
        api_result = client.send_alert_request(alarm_data)
        
        return {
            "status": "success",
            "alarm_no": alarm_no,
            "oss_url": oss_url,
            "api_result": api_result
        }
        
    except Exception as e:
        print(f"[ALERT] 警报处理失败: {e}")
        return {"status": "error", "message": str(e)}


class PersonDetector:
    def __init__(
        self,
        model_name: str | Path = DEFAULT_MODEL_PATH,
        conf: float | None = None,
        iou: float | None = None,
    ) -> None:
        initialize_runtime()

        self.feature_config = FEATURE_CONFIG
        self.model_path = Path(model_name)
        self.model = YOLO(str(self.model_path))
        self.model.overrides["conf"] = DEFAULT_CONF if conf is None else conf
        self.model.overrides["iou"] = DEFAULT_IOU if iou is None else iou
        self.model.overrides["classes"] = [0]

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

        return ImagePrediction(
            image_path=image_path,
            detections=detections,
        )

    def draw_detections(self, frame: Any, prediction: ImagePrediction) -> Any:
        annotated_frame = frame.copy()
        
        for det in prediction.detections:
            x1, y1, x2, y2 = map(int, det.xyxy)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"person {det.confidence:.2f}"
            if det.track_id is not None:
                label_text += f" id:{det.track_id}"
            cv2.putText(annotated_frame, label_text, (x1, max(y1 - 10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if prediction.detections:
            alert_text = f"ALERT: {len(prediction.detections)} person(s) detected!"
            cv2.putText(annotated_frame, alert_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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
        
        annotated_frame = self.draw_detections(frame, prediction)
        
        if prediction.detected:
            send_alert(
                detection_count=len(prediction.detections),
                image_path=str(image_path),
            )
        
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
        window_name: str = "Person Detection",
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

        print(f"实时人员识别已启动 (使用{'ByteTrack跟踪' if use_tracking else '单帧检测'})，摄像头={camera}，按 '{quit_key.upper()}' 退出。")
        frame_count = 0
        failed_reads = 0
        alerted_track_ids: set[int] = set()
        current_track_ids: set[int] = set()
        track_id_timeout: dict[int, int] = {}
        track_timeout_frames = 60

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
                annotated_frame = self.draw_detections(frame, prediction)

                current_track_ids.clear()
                for det in prediction.detections:
                    if det.track_id is not None:
                        current_track_ids.add(det.track_id)
                        if det.track_id not in alerted_track_ids:
                            frame_info = {
                                "frame_number": frame_count,
                                "camera": camera,
                                "person_count": len(prediction.detections),
                                "track_id": det.track_id,
                            }
                            send_alert(
                                detection_count=1,
                                image_path=f"camera:{camera}",
                                frame_info=frame_info,
                                frame=frame,
                            )
                            alerted_track_ids.add(det.track_id)
                            track_id_timeout[det.track_id] = frame_count
                    else:
                        if frame_count % frame_log_interval == 0:
                            print(f"[DETECT] 检测到人员 (无跟踪ID)，已触发警报")
                        send_alert(
                            detection_count=1,
                            image_path=f"camera:{camera}",
                            frame_info={"frame_number": frame_count, "camera": camera},
                            frame=frame,
                        )

                expired_ids = []
                for track_id, last_frame in track_id_timeout.items():
                    if frame_count - last_frame > track_timeout_frames:
                        expired_ids.append(track_id)
                for track_id in expired_ids:
                    if track_id in alerted_track_ids:
                        alerted_track_ids.discard(track_id)
                        print(f"[TRACK] 跟踪ID {track_id} 已超时，移除报警记录")

                if frame_log_interval > 0 and frame_count % frame_log_interval == 0:
                    if prediction.detected:
                        print(f"[DETECT] {len(prediction.detections)} person(s) detected! 已跟踪: {len(alerted_track_ids)} 人")
                    else:
                        print(f"[DETECT] No person detected.")

                cv2.imshow(window_name, annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord(quit_key.lower()):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

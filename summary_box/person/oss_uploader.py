from __future__ import annotations

import datetime
import os
import threading
from pathlib import Path
from typing import Any

from config import cv2


class AlertClient:
    """阿里云 OSS 上传客户端"""

    def __init__(self):
        self.endpoint = os.environ.get("OSS_ENDPOINT", "")
        self.bucket_name = os.environ.get("OSS_BUCKET_NAME", "")
        self.access_key_id = os.environ.get("ALIYUN_ACCESS_KEY_ID", "")
        self.access_key_secret = os.environ.get("ALIYUN_ACCESS_KEY_SECRET", "")

        import oss2
        auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        self.bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)

        self.alarm_counter = 0
        self.counter_lock = threading.Lock()
        self.last_date = datetime.datetime.now().strftime("%Y%m%d")

        self.alert_api_url = os.environ.get("ALERT_API_URL", "")

    def _generate_alarm_no(self) -> str:
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        timestamp_ms = int(datetime.datetime.now().timestamp() * 1000)
        return f"ALARM-{current_date}-{timestamp_ms}"

    def upload_to_oss(self, image_path: str | Path, object_name: str | None = None) -> str:
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
                "appKey": os.environ.get("ALERT_API_APP_KEY", "")
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


def compress_frame(frame: Any, max_width: int = 800, jpeg_quality: int = 60) -> bytes:
    """压缩帧图像，返回 JPEG 字节数据。"""
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return buf.tobytes()

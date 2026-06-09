from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

from config import OUTPUTS_DIR
from .oss_uploader import compress_frame, get_alert_client


def send_alert(
    detection_count: int,
    image_path: str = "",
    frame_info: dict | None = None,
    frame: Any = None,
) -> dict:
    """
    真实警报接口 - 当检测到人员时调用。
    只保存 1 张压缩后的关键帧图片上传 OSS，不存原图。

    参数:
        detection_count: 检测到的人员数量
        image_path: 触发警报的图片路径或来源
        frame_info: 额外的帧信息（用于实时识别时传递）
        frame: 当前帧图像数据（用于实时识别时压缩保存再上传）

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
        is_temp_file = False

        if frame is not None and frame_info:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            camera_name = frame_info.get("camera", "unknown")
            temp_filename = f"person_{timestamp}_{camera_name}.jpg"
            temp_path = OUTPUTS_DIR / temp_filename
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            compressed_bytes = compress_frame(frame, max_width=800, jpeg_quality=60)
            temp_path.write_bytes(compressed_bytes)
            upload_image_path = str(temp_path)
            is_temp_file = True
            print(f"[ALERT] 已保存压缩关键帧 ({len(compressed_bytes) // 1024}KB): {temp_path}")

        if upload_image_path and Path(upload_image_path).exists():
            try:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                object_name = f"person_alert/{timestamp}_{Path(upload_image_path).name}"
                oss_url = client.upload_to_oss(upload_image_path, object_name)
            except Exception as e:
                print(f"[OSS] 上传失败: {e}")
                oss_url = ""
            finally:
                if is_temp_file:
                    try:
                        Path(upload_image_path).unlink()
                        print(f"[ALERT] 已清理临时压缩图片: {upload_image_path}")
                    except Exception as e:
                        print(f"[ALERT] 清理临时图片失败: {e}")

        raw_data = f"confidence\":\"0.93\",\"cameraPoint\":\"{camera_location}"

        alarm_data = {
            "alarmNo": alarm_no,
            "sourceSystem": "IOT_PLATFORM",
            "deviceCode": "2026060001",
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

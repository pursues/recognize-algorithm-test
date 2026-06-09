from __future__ import annotations

import os
from pathlib import Path

from config import cv2, require_cv2


def _get_env(key: str, default: str = "") -> str:
    """读取环境变量，若为空则尝试从 env.conf 兜底加载"""
    val = os.environ.get(key, "")
    if val:
        return val

    # Jetson 等环境下 env.conf 可能未被自动加载，手动兜底
    _conf = Path(__file__).resolve().parent / "env.conf"
    if _conf.exists():
        with open(_conf, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _k, _, _v = _line.partition("=")
                _k = _k.strip()
                _v = _v.strip().strip('"').strip("'")
                if _k == key:
                    return _v
    return default


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


def _is_jetson() -> bool:
    """检测是否运行在 NVIDIA Jetson 平台"""
    try:
        with open("/proc/device-tree/model", "r") as f:
            return "NVIDIA Jetson" in f.read()
    except FileNotFoundError:
        return False


def build_hk_gstreamer_pipeline(
    rtsp_url: str | None = None,
    latency: int = 200,
) -> str:
    if rtsp_url is None:
        rtsp_url = _get_env("HK_RTSP_URL")
    if _is_jetson():
        return (
            f"rtspsrc location={rtsp_url} latency={latency} protocols=tcp ! "
            "rtph264depay ! h264parse ! "
            "nvv4l2decoder ! "
            "nvvidconv ! "
            "video/x-raw,format=BGRx ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink drop=1 max-buffers=2"
        )
    # 非 Jetson 平台使用软件解码
    return (
        f"rtspsrc location={rtsp_url} latency={latency} protocols=tcp ! "
        "rtph264depay ! h264parse ! "
        "avdec_h264 ! "
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
    require_cv2()
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
        rtsp_url = _get_env("HK_RTSP_URL")
        if not rtsp_url:
            raise RuntimeError("未配置 HK_RTSP_URL 环境变量")
        cap = cv2.VideoCapture(rtsp_url)
        for _ in range(5):
            if cap.isOpened():
                break
            cap.release()
            import time
            time.sleep(2)
            cap = cv2.VideoCapture(rtsp_url)
        return cap
    if camera == "usb":
        return cv2.VideoCapture(0, cv2.CAP_ANY)
    if isinstance(camera, str) and camera.isdigit():
        return cv2.VideoCapture(int(camera))
    raise ValueError(f"不支持的摄像头类型: {camera}，可选值为 csi、usb、hk 或数字序号")


def load_rtsp_urls() -> list[str]:
    """从环境变量加载多路 RTSP 地址列表。

    优先读取 HK_RTSP_URLS（逗号分隔），兼容旧的 HK_RTSP_URL（单路）。
    自动跳过空字符串。
    """
    urls_str = _get_env("HK_RTSP_URLS", "")
    if urls_str:
        return [u.strip() for u in urls_str.split(",") if u.strip()]

    # 兼容旧的单路配置
    single_url = _get_env("HK_RTSP_URL", "")
    if single_url:
        return [single_url]

    return []


def open_hk_cameras(
    rtsp_urls: list[str] | None = None,
    width: int = 1280,
    height: int = 720,
) -> list[tuple[str, object]]:
    """打开多路海康 RTSP 摄像头。

    参数:
        rtsp_urls: RTSP 地址列表，为 None 时自动从环境变量加载
        width, height: 目标分辨率

    返回:
        [(camera_name, VideoCapture), ...] 列表
        camera_name 格式为 "hk_0", "hk_1", ...
    """
    require_cv2()

    if rtsp_urls is None:
        rtsp_urls = load_rtsp_urls()

    if not rtsp_urls:
        raise RuntimeError("未配置任何 RTSP 摄像头地址。请在 env.conf 中设置 HK_RTSP_URLS。")

    cameras: list[tuple[str, object]] = []

    for idx, url in enumerate(rtsp_urls):
        cam_name = f"hk_{idx}"
        print(f"正在连接摄像头 {cam_name}: {url} ...")

        cap = None
        for _ in range(5):
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                break
            if cap is not None:
                cap.release()
            import time
            time.sleep(2)

        if cap is None or not cap.isOpened():
            print(f"警告: 摄像头 {cam_name} 连接失败，跳过")
            if cap is not None:
                cap.release()
            continue

        # 设置分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        cameras.append((cam_name, cap))
        print(f"摄像头 {cam_name} 连接成功")

    if not cameras:
        raise RuntimeError(f"所有 {len(rtsp_urls)} 路摄像头均连接失败")

    print(f"共成功连接 {len(cameras)}/{len(rtsp_urls)} 路摄像头")
    return cameras

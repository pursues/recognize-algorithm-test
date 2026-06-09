from __future__ import annotations

from config import cv2


def get_intrusion_zone(width: int, height: int) -> tuple[int, int, int, int]:
    """根据当前帧尺寸等比例缩放闯入区域。"""
    from .config import _BASE_ZONE, _BASE_ZONE_WIDTH, _BASE_ZONE_HEIGHT

    scale_x = width / _BASE_ZONE_WIDTH
    scale_y = height / _BASE_ZONE_HEIGHT
    bx1, by1, bx2, by2 = _BASE_ZONE
    return (
        int(bx1 * scale_x),
        int(by1 * scale_y),
        int(bx2 * scale_x),
        int(by2 * scale_y),
    )


def get_overlap_ratio(det_xyxy: list[float], zone: tuple[int, int, int, int]) -> float:
    """计算单个检测框与闯入区域的重叠比例。

    返回:
        交集面积 / 检测框面积，范围 [0, 1]
    """
    dx1, dy1, dx2, dy2 = det_xyxy
    zx1, zy1, zx2, zy2 = zone

    ix1 = max(dx1, zx1)
    iy1 = max(dy1, zy1)
    ix2 = min(dx2, zx2)
    iy2 = min(dy2, zy2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection_area = (ix2 - ix1) * (iy2 - iy1)
    det_area = (dx2 - dx1) * (dy2 - dy1)

    if det_area <= 0:
        return 0.0

    return intersection_area / det_area


def is_intrusion(detections: list, zone: tuple | None = None, threshold: float | None = None) -> bool:
    """检查是否有检测框与闯入区域的重叠比例达到阈值。

    参数:
        detections: Detection 对象列表
        zone: (x1, y1, x2, y2) 闯入区域坐标，为 None 时使用基准值
        threshold: 重叠比例阈值，为 None 时使用默认值 0.5

    返回:
        True 表示至少有一个检测框满足闯入条件
    """
    from .config import _BASE_ZONE, _INTRUSION_OVERLAP_THRESHOLD

    if zone is None:
        zone = _BASE_ZONE
    if threshold is None:
        threshold = _INTRUSION_OVERLAP_THRESHOLD

    for det in detections:
        if get_overlap_ratio(det.xyxy, zone) >= threshold:
            return True
    return False


def draw_zone_on_frame(frame) -> any:
    """在帧副本上绘制闯入区域红线框并返回。"""
    h, w = frame.shape[:2]
    zx1, zy1, zx2, zy2 = get_intrusion_zone(w, h)
    result = frame.copy()
    cv2.rectangle(result, (zx1, zy1), (zx2, zy2), (0, 0, 255), 2)
    cv2.putText(result, "INTRUSION ZONE", (zx1, max(zy1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return result

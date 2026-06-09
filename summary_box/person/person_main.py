from __future__ import annotations

from typing import Any

from config import Detection

from .alert_sender import send_alert
from .frame_selector import FrameSelector
from .intrusion import draw_zone_on_frame, get_intrusion_zone, get_overlap_ratio, is_intrusion


class PersonIntrusionProcessor:
    """人员闯入算法处理器 — 基于 ByteTrack + 状态机。

    状态机为每个 track_id 维护三种状态：
      OUTSIDE → 未在红框内，不跟踪
      ENTER   → 刚进入红框，内存缓存最佳帧，满足 2/3 重叠 + 积累足够帧后上传 1 次 → INSIDE
      INSIDE  → 在红框内持续移动，完全丢弃后续帧，不重复上报
      LEAVE   → 连续 N 帧消失，释放资源

    上传条件（缺一不可）：
      1. 检测框与红框重叠比例 >= 2/3 (DEEP_INTRUSION_RATIO)
      2. ENTER 状态下积累足够检测帧后，选出置信度最高的帧
      3. 每个 track_id 每次进入红框只上传 1 次
    """

    # 状态常量
    STATE_OUTSIDE = "OUTSIDE"
    STATE_ENTER = "ENTER"
    STATE_INSIDE = "INSIDE"
    STATE_LEAVE = "LEAVE"

    # 身体进入红框 2/3 才算深度闯入
    DEEP_INTRUSION_RATIO = 2.0 / 3.0  # ≈ 0.667

    # ENTER 状态下积累 N 个检测帧后才选出最佳帧
    ENTER_ACCUMULATE_FRAMES = 5

    # 连续 N 帧不在红框内判定为 LEAVE
    LEAVE_ABSENT_FRAMES = 15

    def __init__(self):
        # 每个 track_id 的状态
        self._states: dict[int, str] = {}
        # track_id → 帧缓存器（在 ENTER 状态下选出最佳帧）
        self._frame_cache: dict[int, FrameSelector] = {}
        # track_id → ENTER 状态下看到的检测帧计数
        self._enter_detect_count: dict[int, int] = {}
        # track_id → 连续不在红框内的帧数（LEAVE 判定用）
        self._absent_count: dict[int, int] = {}
        # 无 track_id 时的冷却
        self._no_track_last_alert = 0.0

        self._is_tracking = False

    # ---- 公共接口 ----

    @property
    def is_tracking_mode(self) -> bool:
        """是否有任何 track_id 处于 ENTER 或 INSIDE 状态。"""
        return any(s in (self.STATE_ENTER, self.STATE_INSIDE) for s in self._states.values())

    @property
    def alerted_count(self) -> int:
        return sum(1 for s in self._states.values() if s == self.STATE_INSIDE)

    def reset(self) -> None:
        self._states.clear()
        self._frame_cache.clear()
        self._enter_detect_count.clear()
        self._absent_count.clear()
        self._is_tracking = False
        self._no_track_last_alert = 0.0

    # ---- 每帧处理入口 ----

    def process_frame(
        self,
        detections: list[Detection],
        frame: Any,
        frame_count: int,
        camera: str | int,
    ) -> bool:
        """处理一帧检测结果，返回是否处于跟踪模式。

        不负责任何画面绘制（检测框、红线框均由 main.py 统一绘制）。

        参数:
            detections: 当前帧所有检测结果（仅处理有 track_id 的）
            frame: 当前帧图像（仅用于尺寸计算，不做绘制）
            frame_count: 总帧计数（仅日志用）
            camera: 摄像头标识
        """
        import datetime
        now_time = datetime.datetime.now().timestamp()

        h, w = frame.shape[:2]
        zone = get_intrusion_zone(w, h)

        # 筛选当前帧在红框内的检测（松散阈值 0.5，用于跟踪判断）
        intruding: dict[int, Detection] = {}
        for det in detections:
            if det.track_id is None:
                continue
            if is_intrusion([det], zone):
                intruding[det.track_id] = det

        # ---- 处理当前可见的闯入目标 ----
        for tid, det in intruding.items():
            self._absent_count[tid] = 0  # 重置缺席计数
            current_state = self._states.get(tid, self.STATE_OUTSIDE)

            if current_state == self.STATE_INSIDE:
                # INSIDE → 完全忽略，不做任何处理
                continue

            if current_state == self.STATE_LEAVE:
                # LEAVE → 跳过，等待超时清理
                continue

            # OUTSIDE 或 ENTER
            overlap = get_overlap_ratio(det.xyxy, zone)

            if current_state == self.STATE_OUTSIDE:
                if overlap >= self.DEEP_INTRUSION_RATIO:
                    # 首次深度闯入 → ENTER
                    self._states[tid] = self.STATE_ENTER
                    self._frame_cache[tid] = FrameSelector()
                    self._enter_detect_count[tid] = 1
                    score = det.confidence * ((det.xyxy[2] - det.xyxy[0]) * (det.xyxy[3] - det.xyxy[1]))
                    self._frame_cache[tid].update(tid, frame, score, frame_count)
                    self._is_tracking = True
                    print(f"[PERSON] track_id:{tid} ENTER 红框 (overlap={overlap:.2f}), 开始缓存最佳帧")
                continue

            # STATE_ENTER: 已在 ENTER 状态
            # 满足深度闯入条件才继续缓存，否则重置为 OUTSIDE
            if overlap < self.DEEP_INTRUSION_RATIO:
                # 还没达到 2/3，可能刚碰到边 → 重置为 OUTSIDE
                self._cleanup_track(tid)
                continue

            # 深度闯入 → 缓存帧，计数
            self._enter_detect_count[tid] = self._enter_detect_count.get(tid, 0) + 1
            score = det.confidence * ((det.xyxy[2] - det.xyxy[0]) * (det.xyxy[3] - det.xyxy[1]))
            self._frame_cache[tid].update(tid, frame, score, frame_count)

            # 积累足够帧数 → 选最佳帧上传 → 转入 INSIDE
            if self._enter_detect_count[tid] >= self.ENTER_ACCUMULATE_FRAMES:
                best = self._frame_cache[tid].get_best_frame(tid)
                if best is not None:
                    print(f"[PERSON] track_id:{tid} 积累 {self._enter_detect_count[tid]} 帧, 上传最佳帧 → INSIDE")
                    send_alert(
                        detection_count=1,
                        image_path=f"camera:{camera}",
                        frame_info={
                            "frame_number": frame_count,
                            "camera": camera,
                            "track_id": tid,
                            "overlap_ratio": round(overlap, 2),
                        },
                        frame=draw_zone_on_frame(best),
                    )
                self._states[tid] = self.STATE_INSIDE
                self._frame_cache.pop(tid, None)
                self._enter_detect_count.pop(tid, None)

        # ---- 处理不在红框内的已跟踪目标（缺席计数） ----
        tracked_ids = [tid for tid, s in self._states.items() if s in (self.STATE_ENTER, self.STATE_INSIDE)]
        for tid in tracked_ids:
            if tid not in intruding:
                self._absent_count[tid] = self._absent_count.get(tid, 0) + 1
                if self._absent_count[tid] >= self.LEAVE_ABSENT_FRAMES:
                    print(f"[PERSON] track_id:{tid} 连续 {self._absent_count[tid]} 帧不在红框 → LEAVE")
                    self._cleanup_track(tid)

        # ---- 无 track_id 的后备逻辑（ByteTrack 未稳定时） ----
        # 必须满足 2/3 深度闯入 + 冷却 10s 才上传，防止跟踪器初始化阶段误报
        no_track_dets = [d for d in detections
                         if d.track_id is None
                         and get_overlap_ratio(d.xyxy, zone) >= self.DEEP_INTRUSION_RATIO]
        if no_track_dets and (now_time - self._no_track_last_alert) >= 10.0:
            self._no_track_last_alert = now_time
            print(f"[PERSON] 无跟踪ID深度闯入，上传当前帧")
            send_alert(
                detection_count=len(no_track_dets),
                image_path=f"camera:{camera}",
                frame_info={"frame_number": frame_count, "camera": camera},
                frame=draw_zone_on_frame(frame),
            )

        return self.is_tracking_mode

    # ---- 内部 ----

    def _cleanup_track(self, tid: int) -> None:
        self._states.pop(tid, None)
        self._frame_cache.pop(tid, None)
        self._enter_detect_count.pop(tid, None)
        self._absent_count.pop(tid, None)

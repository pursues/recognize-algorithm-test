from __future__ import annotations

from typing import Any


class FrameSelector:
    """独立管理每个跟踪目标的最佳帧选取逻辑。

    每个 track_id 维护自己独立的：
    - best_frame: 截止目前分数最高的帧
    - best_score: 最高分数
    - first_detect_frame: 首次检测帧号（用于延迟上传）
    """

    def __init__(self, upload_delay: int = 3):
        self.upload_delay = upload_delay              # 等待 N 个检测帧后再上传
        self._best_frames: dict[int, Any] = {}         # track_id → 最佳帧
        self._best_scores: dict[int, float] = {}       # track_id → 最高 score
        self._first_frames: dict[int, int] = {}         # track_id → 首次检测帧号

    def update(self, track_id: int, frame: Any, score: float, frame_count: int) -> None:
        """用当前帧更新指定目标的最佳帧候选。

        参数:
            track_id: ByteTrack 跟踪 ID
            frame: 当前帧的副本
            score: 当前帧该检测框的分数 (confidence * area)
            frame_count: 当前总帧数计数
        """
        if track_id not in self._first_frames:
            self._first_frames[track_id] = frame_count

        prev_score = self._best_scores.get(track_id, 0.0)
        if score > prev_score:
            self._best_scores[track_id] = score
            self._best_frames[track_id] = frame.copy()

    def is_ready(self, track_id: int, frame_count: int, skip_frames: int) -> bool:
        """检查目标是否积累了足够的检测帧，可以上传。"""
        first_frame = self._first_frames.get(track_id)
        if first_frame is None:
            return False
        detect_frames_elapsed = (frame_count - first_frame) // max(skip_frames, 1)
        return detect_frames_elapsed >= self.upload_delay

    def reset_for_track_id(self, track_id: int) -> None:
        """上传后重置该目标的帧选择状态，允许重新积累。"""
        self._best_frames.pop(track_id, None)
        self._best_scores.pop(track_id, None)
        self._first_frames.pop(track_id, None)

    def get_best_frame(self, track_id: int) -> Any | None:
        """获取目标的最佳帧。"""
        return self._best_frames.get(track_id)

    def remove(self, track_id: int) -> None:
        """移除目标的所有记录。"""
        self._best_frames.pop(track_id, None)
        self._best_scores.pop(track_id, None)
        self._first_frames.pop(track_id, None)

    def clear(self) -> None:
        """清空所有记录。"""
        self._best_frames.clear()
        self._best_scores.clear()
        self._first_frames.clear()

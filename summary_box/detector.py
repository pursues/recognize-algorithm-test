from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO

from config import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    DEFAULT_MODEL_PATH,
    Detection,
    ImagePrediction,
    cv2,
    initialize_runtime,
    require_cv2,
)


class PersonDetector:
    """通用检测器 - 使用 YOLOv8n 模型进行人员检测和 ByteTrack 跟踪

    只负责模型推理和检测结果构建，不包含任何业务逻辑（闯入判断、报警等）。
    """

    def __init__(
        self,
        model_name: str | Path = DEFAULT_MODEL_PATH,
        conf: float | None = None,
        iou: float | None = None,
    ) -> None:
        initialize_runtime()

        self.model_path = Path(model_name)
        self.model = YOLO(str(self.model_path))
        self.model.overrides["conf"] = DEFAULT_CONF if conf is None else conf
        self.model.overrides["iou"] = DEFAULT_IOU if iou is None else iou
        self.model.overrides["classes"] = [0]  # COCO person class

    @property
    def class_names(self) -> dict[int, str]:
        return self.model.model.names

    def build_prediction(self, result: Any, image_path: Path) -> ImagePrediction:
        """从模型结果构建 ImagePrediction"""
        boxes = result.boxes
        detections: list[Detection] = []

        if boxes is not None:
            for index in range(len(boxes)):
                cls_id = int(boxes.cls[index].item())
                label = self.class_names.get(cls_id, str(cls_id))

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
        """在帧上绘制检测框和标签"""
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
        """对单张图片进行人员检测"""
        image_path = Path(image_path)
        require_cv2()
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

        prediction = self.build_prediction(results[0], image_path=image_path)
        annotated_frame = self.draw_detections(frame, prediction)

        if save and output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / image_path.name
            cv2.imwrite(str(save_path), annotated_frame)

        return prediction, annotated_frame

    def track_or_detect(
        self,
        frame: Any,
        conf: float | None = None,
        iou: float | None = None,
        verbose: bool = False,
    ):
        """跟踪或检测单帧，返回原始 ultralytics Results。

        优先使用 track()（保持 ByteTrack 状态连续），失败时降级为 predict()。
        """
        try:
            return self.model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=self.model.overrides["conf"] if conf is None else conf,
                iou=self.model.overrides["iou"] if iou is None else iou,
                verbose=verbose,
            )
        except (ImportError, ModuleNotFoundError):
            return self.model.predict(
                source=frame,
                conf=self.model.overrides["conf"] if conf is None else conf,
                iou=self.model.overrides["iou"] if iou is None else iou,
                verbose=verbose,
            )

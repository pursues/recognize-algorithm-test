from __future__ import annotations

import argparse
import json
from pathlib import Path

from play_phone import DEFAULT_IMAGE_PATH
from play_phone import DEFAULT_MODEL_NAME
from play_phone import FEATURE_CONFIG
from play_phone import OUTPUTS_DIR
from play_phone import PlayPhoneDetector
from play_phone import TEST_IMAGES_DIR
from play_phone import save_prediction_visualization
from play_phone import summarize_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="测试 play_phone 目录下的玩手机识别能力。")
    parser.add_argument("--image", type=Path, help="待检测图片路径；如果未提供，则默认处理整个图片目录")
    parser.add_argument("--image-dir", type=Path, default=TEST_IMAGES_DIR, help="待批量检测的图片目录")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="模型名称或本地权重路径；默认使用开源预训练权重 yolov8n.pt",
    )
    parser.add_argument("--conf", type=float, help="置信度阈值")
    parser.add_argument("--iou", type=float, help="NMS IoU 阈值")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR,
        help="结果输出目录，包含 result.json 与可视化结果",
    )
    parser.add_argument(
        "--no-save-vis",
        action="store_true",
        help="不保存可视化结果",
    )
    parser.add_argument(
        "--default-image",
        action="store_true",
        help="强制仅测试默认示例图",
    )
    return parser


def _serialize_prediction(prediction, vis_path: Path | None) -> dict[str, object]:
    return {
        "image": str(prediction.image_path),
        "detected": prediction.detected,
        "face_boxes": prediction.face_boxes,
        "detections": [
            {
                "label": detection.label,
                "confidence": round(detection.confidence, 4),
                "xyxy": [round(value, 2) for value in detection.xyxy],
            }
            for detection in prediction.detections
        ],
        "play_phone_events": [
            {
                "person_xyxy": event.person_xyxy,
                "phone_xyxy": event.phone_xyxy,
                "face_xyxy": event.face_xyxy,
                "score": event.score,
                "holding_phone": event.holding_phone,
                "looking_phone": event.looking_phone,
                "matched_rules": event.matched_rules,
            }
            for event in prediction.phone_usage_events
        ],
        "vis_path": str(vis_path) if vis_path is not None else None,
    }


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = PlayPhoneDetector(
        model_name=str(args.model),
        conf=args.conf,
        iou=args.iou,
    )

    if args.default_image:
        image_path = DEFAULT_IMAGE_PATH.resolve()
        predictions = [detector.predict_image(image_path)]
    elif args.image is not None:
        image_path = args.image.resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"测试图片不存在: {image_path}")
        predictions = [detector.predict_image(image_path)]
    else:
        image_dir = args.image_dir.resolve()
        predictions = detector.predict_directory(image_dir)

    serialized_predictions: list[dict[str, object]] = []
    for prediction in predictions:
        vis_path = None
        if not args.no_save_vis:
            vis_path = save_prediction_visualization(
                prediction=prediction,
                output_dir=output_dir,
                name="predict",
            )
        serialized_predictions.append(_serialize_prediction(prediction, vis_path))

    result = {
        "feature": FEATURE_CONFIG.name,
        "display_name": FEATURE_CONFIG.display_name,
        "expected_labels": list(FEATURE_CONFIG.expected_labels),
        "model": str(args.model),
        "save_vis": not args.no_save_vis,
        "output_dir": str(output_dir),
        "summary": summarize_predictions(predictions),
        "items": serialized_predictions,
    }
    result_file = output_dir / "result.json"
    result_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result["result_file"] = str(result_file)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

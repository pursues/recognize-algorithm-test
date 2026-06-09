from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import (
    DEFAULT_MODEL_PATH,
    OUTPUTS_DIR,
    TEST_IMAGES_DIR,
)
from detector import PersonDetector
from person import FEATURE_CONFIG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="测试 person 目录下的人员识别模型。")
    parser.add_argument("--image", type=Path, help="待检测图片路径。如果不提供，将尝试测试 imgs 目录下的图片。")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="模型路径")
    parser.add_argument("--conf", type=float, help="置信度阈值")
    parser.add_argument("--iou", type=float, help="NMS IoU 阈值")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR,
        help="可视化结果输出目录",
    )
    parser.add_argument(
        "--no-save-vis",
        action="store_true",
        help="不保存带检测框的可视化结果",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_path = args.model.resolve()

    detector = PersonDetector(
        model_name=str(model_path),
        conf=args.conf,
        iou=args.iou,
    )

    image_paths = []
    if args.image:
        image_paths.append(args.image.resolve())
    else:
        if TEST_IMAGES_DIR.exists():
            for ext in [".jpg", ".jpeg", ".png"]:
                image_paths.extend(list(TEST_IMAGES_DIR.glob(f"*{ext}")))

    if not image_paths:
        raise FileNotFoundError(f"未找到测试图片。请使用 --image 指定图片路径，或在 {TEST_IMAGES_DIR} 中放置图片。")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for image_path in image_paths:
        prediction, _ = detector.predict_image(
            image_path=image_path,
            save=not args.no_save_vis,
            output_dir=output_dir if not args.no_save_vis else None,
        )

        result = {
            "feature": FEATURE_CONFIG.name,
            "display_name": FEATURE_CONFIG.display_name,
            "model": str(model_path),
            "image": str(image_path),
            "expected_labels": list(FEATURE_CONFIG.expected_labels),
            "person_count": len(prediction.detections),
            "has_person": prediction.detected,
            "detections": [
                {
                    "label": detection.label,
                    "confidence": round(detection.confidence, 4),
                    "xyxy": [round(value, 2) for value in detection.xyxy],
                    "track_id": detection.track_id,
                }
                for detection in prediction.detections
            ],
            "save_vis": not args.no_save_vis,
            "output_dir": str(output_dir),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    result_file = output_dir / "result.json"
    result_file.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n详细结果已保存至 {result_file}")


if __name__ == "__main__":
    main()

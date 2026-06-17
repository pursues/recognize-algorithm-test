from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import (
    BEHAVIOR_CONFIG,
    DEFAULT_MODEL_PATH,
    IMAGE_SUFFIXES,
    OUTPUTS_DIR,
    TEST_IMAGES_DIR,
    classify_behaviors,
    cv2,
    initialize_runtime,
)
from detector import PersonDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="行为识别测试入口 - 使用 best.pt 模型对 imgs 下图片进行测试"
    )
    parser.add_argument("--image", type=Path, help="单独测试一张图片")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="模型路径")
    parser.add_argument("--conf", type=float, help="置信度阈值")
    parser.add_argument("--iou", type=float, help="NMS IoU 阈值")
    parser.add_argument("--no-save-vis", action="store_true", help="不保存带检测框的可视化结果")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR, help="输出根目录")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    initialize_runtime()

    model_path = args.model.resolve()
    print(f"加载模型: {model_path}")

    detector = PersonDetector(
        model_name=str(model_path),
        conf=args.conf,
        iou=args.iou,
    )
    print(f"模型类别: {detector.class_names}")

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    save_vis = not args.no_save_vis

    image_paths = []
    if args.image:
        image_paths.append(args.image.resolve())
    elif TEST_IMAGES_DIR.exists():
        for ext in IMAGE_SUFFIXES:
            image_paths.extend(sorted(TEST_IMAGES_DIR.glob(f"*{ext}")))

    if not image_paths:
        raise FileNotFoundError(f"未找到测试图片。使用 --image 指定路径，或在 {TEST_IMAGES_DIR} 放置图片。")

    results = []
    for image_path in image_paths:
        prediction, _ = detector.predict_image(
            image_path=image_path,
            save=save_vis,
            output_dir=output_root if save_vis else None,
        )

        person_dets = [d for d in prediction.detections
                       if d.label.lower() in ("person",)]
        behavior_dets = [d for d in prediction.detections
                         if d.label.lower() not in ("person",)]
        behaviors = classify_behaviors(prediction.detections)
        detected_labels = list(set(d.label for d in prediction.detections))

        print(f"\n{'='*60}")
        print(f"图片: {image_path.name}")
        print(f"{'='*60}")
        print(f"  检测到 {len(person_dets)} 人 / {len(behavior_dets)} 个行为目标")

        if behaviors:
            print(f"  匹配的行为:")
            for bname, bdets in behaviors.items():
                binfo = BEHAVIOR_CONFIG.get(bname, {})
                blabels = list(set(d.label for d in bdets))
                print(f"    - {binfo.get('display_name', bname)}: {blabels} (共{len(bdets)}个)")
        elif prediction.detected:
            print(f"  检测到但未匹配已知行为: {detected_labels}")
        else:
            print(f"  未检测到任何目标")

        result = {
            "model": str(model_path),
            "image": str(image_path),
            "person_count": len(person_dets),
            "behavior_count": len(behavior_dets),
            "detected_labels": detected_labels,
            "total_detections": len(prediction.detections),
            "has_detection": prediction.detected,
            "matched_behaviors": list(behaviors.keys()) if behaviors else [],
            "detections": [
                {
                    "label": d.label,
                    "confidence": round(d.confidence, 4),
                    "xyxy": [round(v, 2) for v in d.xyxy],
                    "track_id": d.track_id,
                }
                for d in prediction.detections
            ],
        }
        results.append(result)

    total = len(results)
    detected_count = sum(1 for r in results if r["has_detection"])
    with_behavior = sum(1 for r in results if r["matched_behaviors"])
    print(f"\n汇总: 总数={total} | 有检测={detected_count} | 匹配行为={with_behavior}")

    result_file = output_root / "result.json"
    result_file.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"详细结果已保存至 {result_file}")


if __name__ == "__main__":
    main()

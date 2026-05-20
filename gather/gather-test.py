from __future__ import annotations

import argparse
import json
from pathlib import Path

from gather import DEFAULT_MODEL_PATH
from gather import FEATURE_CONFIG
from gather import OUTPUTS_DIR
from gather import TEST_IMAGES_DIR
from gather import GatherDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="测试 gather 目录下的人员聚集识别模型。")
    parser.add_argument("--image", type=Path, help="待检测图片路径。如果不提供，将尝试测试 imgs 目录下的图片。")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="模型路径")
    parser.add_argument(
        "--camera",
        help="实时识别摄像头来源，可选 csi、usb 或数字序号，例如 0",
    )
    parser.add_argument("--conf", type=float, help="置信度阈值")
    parser.add_argument("--iou", type=float, help="NMS IoU 阈值")
    parser.add_argument("--distance-threshold", type=float, default=150.0, help="判定聚集的像素距离阈值")
    parser.add_argument("--min-gather-count", type=int, default=3, help="判定聚集的最少人数")
    parser.add_argument("--frame-log-interval", type=int, default=10, help="每隔多少帧打印一次检测结果")
    parser.add_argument("--width", type=int, default=1280, help="CSI 摄像头宽度")
    parser.add_argument("--height", type=int, default=720, help="CSI 摄像头高度")
    parser.add_argument("--framerate", type=int, default=30, help="CSI 摄像头帧率")
    parser.add_argument("--flip-method", type=int, default=0, help="CSI 摄像头翻转方式")
    parser.add_argument("--warmup-frames", type=int, default=5, help="摄像头预热读取帧数")
    parser.add_argument("--max-failed-reads", type=int, default=30, help="连续读帧失败多少次后终止")
    parser.add_argument("--window-name", default="Gather Detection", help="实时识别窗口标题")
    parser.add_argument("--use-tracking", action="store_true", help="启用 ByteTrack 跟踪算法（默认在摄像头模式下启用）")
    parser.add_argument("--no-tracking", action="store_true", help="禁用 ByteTrack 跟踪算法")
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
    
    detector = GatherDetector(
        model_name=str(model_path),
        conf=args.conf,
        iou=args.iou,
        distance_threshold=args.distance_threshold,
        min_gather_count=args.min_gather_count,
    )

    if args.camera:
        use_tracking = True
        if args.no_tracking:
            use_tracking = False
            
        detector.run_camera_inference(
            camera=args.camera,
            conf=args.conf,
            iou=args.iou,
            frame_log_interval=args.frame_log_interval,
            window_name=args.window_name,
            width=args.width,
            height=args.height,
            framerate=args.framerate,
            flip_method=args.flip_method,
            warmup_frames=args.warmup_frames,
            max_failed_reads=args.max_failed_reads,
            use_tracking=use_tracking,
        )
        return

    # 尝试查找 imgs 目录下的测试图片
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
    
    use_tracking = args.use_tracking

    results = []
    for image_path in image_paths:
        prediction, _ = detector.predict_image(
            image_path=image_path,
            save=not args.no_save_vis,
            output_dir=output_dir if not args.no_save_vis else None,
            use_tracking=use_tracking,
        )

        result = {
            "feature": FEATURE_CONFIG.name,
            "display_name": FEATURE_CONFIG.display_name,
            "model": str(model_path),
            "image": str(image_path),
            "expected_labels": list(FEATURE_CONFIG.expected_labels),
            "people_count": len(prediction.detections),
            "gather_groups_count": len(prediction.gather_groups),
            "has_gathering": prediction.has_gathering,
            "detections": [
                {
                    "label": detection.label,
                    "confidence": round(detection.confidence, 4),
                    "xyxy": [round(value, 2) for value in detection.xyxy],
                    "track_id": detection.track_id,
                }
                for detection in prediction.detections
            ],
            "gather_groups": [
                {
                    "group_id": group.group_id,
                    "people_count": len(group.detections),
                    "detections": [
                        {
                            "label": det.label,
                            "confidence": round(det.confidence, 4),
                            "xyxy": [round(value, 2) for value in det.xyxy],
                            "track_id": det.track_id,
                        }
                        for det in group.detections
                    ]
                }
                for group in prediction.gather_groups
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
    print(f"\\n详细结果已保存至 {result_file}")


if __name__ == "__main__":
    main()

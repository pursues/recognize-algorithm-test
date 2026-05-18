from __future__ import annotations

import argparse
import json
from pathlib import Path


from helmet import DEFAULT_IMAGE_PATH
from helmet import DEFAULT_MODEL_PATH
from helmet import FEATURE_CONFIG
from helmet import OUTPUTS_DIR
from helmet import HelmetDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="测试 helmet 目录下的安全帽识别模型。")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH, help="待检测图片路径")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="模型路径")
    parser.add_argument(
        "--camera",
        help="实时识别摄像头来源，可选 csi、usb 或数字序号，例如 0",
    )
    parser.add_argument("--conf", type=float, help="置信度阈值")
    parser.add_argument("--iou", type=float, help="NMS IoU 阈值")
    parser.add_argument("--frame-log-interval", type=int, default=10, help="每隔多少帧打印一次检测结果")
    parser.add_argument("--width", type=int, default=1280, help="CSI 摄像头宽度")
    parser.add_argument("--height", type=int, default=720, help="CSI 摄像头高度")
    parser.add_argument("--framerate", type=int, default=30, help="CSI 摄像头帧率")
    parser.add_argument("--flip-method", type=int, default=0, help="CSI 摄像头翻转方式")
    parser.add_argument("--warmup-frames", type=int, default=5, help="摄像头预热读取帧数")
    parser.add_argument("--max-failed-reads", type=int, default=30, help="连续读帧失败多少次后终止")
    parser.add_argument("--window-name", default="Helmet Detection", help="实时识别窗口标题")
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
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    detector = HelmetDetector(
        model_name=str(model_path),
        conf=args.conf,
        iou=args.iou,
    )

    if args.camera:
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
        )
        return

    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"测试图片不存在: {image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    prediction = detector.predict_image(
        image_path=image_path,
        save=not args.no_save_vis,
        project=output_dir if not args.no_save_vis else None,
        name="predict",
    )

    result = {
        "feature": FEATURE_CONFIG.name,
        "display_name": FEATURE_CONFIG.display_name,
        "model": str(model_path),
        "image": str(image_path),
        "detections": [
            {
                "label": detection.label,
                "confidence": round(detection.confidence, 4),
                "xyxy": [round(value, 2) for value in detection.xyxy],
            }
            for detection in prediction.detections
        ],
        "detected": prediction.detected,
        "save_vis": not args.no_save_vis,
        "output_dir": str(output_dir),
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

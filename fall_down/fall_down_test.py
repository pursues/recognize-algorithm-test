from __future__ import annotations

import argparse
import json
from pathlib import Path
import cv2

from fall_down import DEFAULT_MODEL_NAME
from fall_down import FEATURE_CONFIG
from fall_down import OUTPUTS_DIR
from fall_down import TEST_IMAGES_DIR
from fall_down import FallDownDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="测试 fall_down 目录下的摔倒识别模型。")
    parser.add_argument("--image", type=Path, help="待检测图片路径（可选）")
    parser.add_argument("--dir", type=Path, help="待检测图片目录路径（可选）")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="模型路径或名称，默认为 yolov8n.pt")
    parser.add_argument(
        "--camera",
        help="实时识别摄像头来源，可选 csi、usb 或数字序号，例如 0",
    )
    parser.add_argument("--conf", type=float, help="置信度阈值")
    parser.add_argument("--iou", type=float, help="NMS IoU 阈值")
    parser.add_argument("--fall-threshold", type=float, default=1.0, help="判定为摔倒的边界框宽高比阈值")
    parser.add_argument("--frame-log-interval", type=int, default=10, help="每隔多少帧打印一次检测结果")
    parser.add_argument("--width", type=int, default=1280, help="CSI 摄像头宽度")
    parser.add_argument("--height", type=int, default=720, help="CSI 摄像头高度")
    parser.add_argument("--framerate", type=int, default=30, help="CSI 摄像头帧率")
    parser.add_argument("--flip-method", type=int, default=0, help="CSI 摄像头翻转方式")
    parser.add_argument("--warmup-frames", type=int, default=5, help="摄像头预热读取帧数")
    parser.add_argument("--max-failed-reads", type=int, default=30, help="连续读帧失败多少次后终止")
    parser.add_argument("--window-name", default="Fall Down Detection", help="实时识别窗口标题")
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


def process_image(detector: FallDownDetector, image_path: Path, output_dir: Path, save_vis: bool):
    prediction, result = detector.predict_image(
        image_path=image_path,
        save=False,
        name="predict",
    )
    
    fall_detections = [
        detection for detection in prediction.detections if detection.is_fall
    ]

    result_dict = {
        "feature": FEATURE_CONFIG.name,
        "display_name": FEATURE_CONFIG.display_name,
        "model": str(detector.model_path),
        "image": str(image_path),
        "detections": [
            {
                "label": detection.label,
                "confidence": round(detection.confidence, 4),
                "xyxy": [round(value, 2) for value in detection.xyxy],
                "is_fall": detection.is_fall
            }
            for detection in prediction.detections
        ],
        "fall_detected": bool(fall_detections),
        "save_vis": save_vis,
        "output_dir": str(output_dir),
    }
    
    if save_vis:
        img = cv2.imread(str(image_path))
        annotated_img = detector._plot_custom(result, prediction, img)
        out_path = output_dir / image_path.name
        cv2.imwrite(str(out_path), annotated_img)
        result_dict["vis_path"] = str(out_path)

    return result_dict


def main() -> None:
    args = build_parser().parse_args()
    
    detector = FallDownDetector(
        model_name=args.model,
        conf=args.conf,
        iou=args.iou,
        fall_threshold=args.fall_threshold,
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

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    if args.dir:
        dir_path = args.dir.resolve()
        if not dir_path.exists():
            raise FileNotFoundError(f"测试目录不存在: {dir_path}")
        from fall_down import iter_image_files
        for img_path in iter_image_files(dir_path):
            res = process_image(detector, img_path, output_dir, not args.no_save_vis)
            results.append(res)
    elif args.image:
        image_path = args.image.resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"测试图片不存在: {image_path}")
        res = process_image(detector, image_path, output_dir, not args.no_save_vis)
        results.append(res)
    else:
        # Default to processing the imgs directory
        dir_path = TEST_IMAGES_DIR.resolve()
        if dir_path.exists():
            from fall_down import iter_image_files
            for img_path in iter_image_files(dir_path):
                res = process_image(detector, img_path, output_dir, not args.no_save_vis)
                results.append(res)
        else:
            print(f"默认测试目录不存在: {dir_path}，请指定 --image 或 --dir")
            return

    result_file = output_dir / "result.json"
    result_file.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

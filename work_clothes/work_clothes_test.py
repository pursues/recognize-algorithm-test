from __future__ import annotations

import argparse
import json
from pathlib import Path

from work_clothes import DEFAULT_CONF
from work_clothes import DEFAULT_IMAGE_PATH
from work_clothes import DEFAULT_IOU
from work_clothes import DEFAULT_PPE_MODEL_PATH
from work_clothes import FEATURE_CONFIG
from work_clothes import OUTPUTS_DIR
from work_clothes import WorkClothesDetector
from work_clothes import save_prediction_visualization


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="测试 work_clothes 目录下的工作服识别模型。")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH, help="待检测图片路径")
    parser.add_argument("--image-dir", type=Path, help="批量检测图片目录")
    parser.add_argument(
        "--camera",
        nargs="?",
        const="csi",
        help="实时识别摄像头来源；不带值时默认使用 csi，可选 csi、usb 或数字序号，例如 0",
    )
    parser.add_argument(
        "--ppe-model",
        type=Path,
        default=DEFAULT_PPE_MODEL_PATH,
        help="已训练好的安全马甲/PPE 权重路径",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help="检测置信度阈值",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU,
        help="NMS IoU 阈值",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR,
        help="结果输出目录",
    )
    parser.add_argument(
        "--no-save-vis",
        action="store_true",
        help="不保存带框可视化结果",
    )
    parser.add_argument("--frame-log-interval", type=int, default=10, help="每隔多少帧打印一次检测结果")
    parser.add_argument("--width", type=int, default=1280, help="CSI 摄像头宽度")
    parser.add_argument("--height", type=int, default=720, help="CSI 摄像头高度")
    parser.add_argument("--framerate", type=int, default=30, help="CSI 摄像头帧率")
    parser.add_argument("--flip-method", type=int, default=0, help="CSI 摄像头翻转方式")
    parser.add_argument("--warmup-frames", type=int, default=5, help="摄像头预热读取帧数")
    parser.add_argument("--max-failed-reads", type=int, default=30, help="连续读帧失败多少次后终止")
    parser.add_argument("--window-name", default="Safety Vest Detection", help="实时识别窗口标题")
    return parser


def _prediction_to_dict(prediction, save_vis: bool, output_dir: Path) -> dict:
    work_clothes_detections = [
        {
            "label": detection.label,
            "raw_class_name": detection.raw_class_name,
            "confidence": round(detection.confidence, 4),
            "xyxy": [round(value, 2) for value in detection.xyxy],
            "status": detection.status,
            "is_work_clothes": detection.is_work_clothes,
        }
        for detection in prediction.detections
        if detection.is_work_clothes
    ]
    detections = [
        {
            "label": detection.label,
            "raw_class_name": detection.raw_class_name,
            "confidence": round(detection.confidence, 4),
            "xyxy": [round(value, 2) for value in detection.xyxy],
            "status": detection.status,
            "is_work_clothes": detection.is_work_clothes,
        }
        for detection in prediction.detections
    ]

    result = {
        "feature": FEATURE_CONFIG.name,
        "display_name": FEATURE_CONFIG.display_name,
        "image": str(prediction.image_path),
        "positive_labels": list(FEATURE_CONFIG.positive_labels),
        "negative_labels": list(FEATURE_CONFIG.negative_labels),
        "detections": detections,
        "work_clothes_detections": work_clothes_detections,
        "detected": bool(work_clothes_detections),
        "save_vis": save_vis,
        "output_dir": str(output_dir),
    }
    return result


def main() -> None:
    args = build_parser().parse_args()
    ppe_model_path = args.ppe_model.resolve()
    if not ppe_model_path.exists():
        raise FileNotFoundError(f"PPE 权重不存在: {ppe_model_path}")

    detector = WorkClothesDetector(
        ppe_model_name=str(ppe_model_path),
        conf=args.conf,
        iou=args.iou,
    )

    if args.camera is not None:
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

    if args.image_dir:
        image_dir = args.image_dir.resolve()
        predictions = detector.predict_directory(image_dir)
        results = []
        for prediction in predictions:
            if not args.no_save_vis:
                save_prediction_visualization(prediction, output_dir=output_dir, name="predict")
            results.append(_prediction_to_dict(prediction, save_vis=not args.no_save_vis, output_dir=output_dir))
        result = {"items": results}
    else:
        image_path = args.image.resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"测试图片不存在: {image_path}")
        prediction = detector.predict_image(image_path)
        if not args.no_save_vis:
            save_prediction_visualization(prediction, output_dir=output_dir, name="predict")
        result = _prediction_to_dict(prediction, save_vis=not args.no_save_vis, output_dir=output_dir)

    result_file = output_dir / "result.json"
    result_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if isinstance(result, dict):
        result["result_file"] = str(result_file)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

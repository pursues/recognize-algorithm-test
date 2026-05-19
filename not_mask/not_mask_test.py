from __future__ import annotations

import argparse
import json
from pathlib import Path

from not_mask import DEFAULT_IMAGE_PATH
from not_mask import DEFAULT_MIN_FACE_SIZE
from not_mask import DEFAULT_MODEL_NAME
from not_mask import DEFAULT_SCORE_THRESHOLD
from not_mask import FEATURE_CONFIG
from not_mask import OUTPUTS_DIR
from not_mask import TEST_IMAGES_DIR
from not_mask import NotMaskDetector
from not_mask import save_prediction_visualization
from not_mask import summarize_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="测试 not_mask 目录下的未戴口罩识别能力。")
    parser.add_argument(
        "--image",
        type=Path,
        help="待检测图片路径；如果未提供，则默认处理整个图片目录",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=TEST_IMAGES_DIR,
        help="待批量检测的图片目录",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="本地模型目录；默认使用当前目录下的本地模型",
    )
    parser.add_argument(
        "--camera",
        help="实时识别摄像头来源，可选 csi、usb 或数字序号，例如 0",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help="判定为未戴口罩的最低置信度阈值",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=DEFAULT_MIN_FACE_SIZE,
        help="OpenCV 人脸检测的最小人脸尺寸",
    )
    parser.add_argument("--frame-log-interval", type=int, default=10, help="每隔多少帧打印一次检测结果")
    parser.add_argument("--width", type=int, default=1280, help="CSI 摄像头宽度")
    parser.add_argument("--height", type=int, default=720, help="CSI 摄像头高度")
    parser.add_argument("--framerate", type=int, default=30, help="CSI 摄像头帧率")
    parser.add_argument("--flip-method", type=int, default=0, help="CSI 摄像头翻转方式")
    parser.add_argument("--warmup-frames", type=int, default=5, help="摄像头预热读取帧数")
    parser.add_argument("--max-failed-reads", type=int, default=30, help="连续读帧失败多少次后终止")
    parser.add_argument("--window-name", default="Not Mask Detection", help="实时识别窗口标题")
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
        "face_detections": [
            {
                "label": detection.label,
                "confidence": detection.confidence,
                "xyxy": detection.xyxy,
                "is_no_mask": detection.is_no_mask,
                "source": detection.source,
                "label_scores": detection.label_scores,
            }
            for detection in prediction.face_detections
        ],
        "vis_path": str(vis_path) if vis_path is not None else None,
    }


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = NotMaskDetector(
        model_name=str(args.model),
        score_threshold=args.score_threshold,
        min_face_size=args.min_face_size,
    )

    if args.camera:
        detector.run_camera_inference(
            camera=args.camera,
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

    if args.default_image:
        predictions = [detector.predict_image(DEFAULT_IMAGE_PATH.resolve())]
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
        "score_threshold": args.score_threshold,
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

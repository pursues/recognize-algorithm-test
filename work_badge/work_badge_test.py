from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORK_BADGE_DIR = Path(__file__).resolve().parent

from work_badge import DEFAULT_BOX_THRESHOLD
from work_badge import DEFAULT_CONF
from work_badge import DEFAULT_IMAGE_PATH
from work_badge import DEFAULT_IOU
from work_badge import DEFAULT_PERSON_MODEL_PATH
from work_badge import DEFAULT_TEXT_THRESHOLD
from work_badge import DEFAULT_ZERO_SHOT_MODEL_NAME
from work_badge import DEFAULT_ZERO_SHOT_MODEL_PATH
from work_badge import DEFAULT_ZERO_SHOT_MODEL_REPO_ID
from work_badge import FEATURE_CONFIG
from work_badge import OUTPUTS_DIR
from work_badge import WorkBadgeDetector
from work_badge import ensure_local_zero_shot_model
from work_badge import save_prediction_visualization
from work_badge import summarize_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="测试 work_badge 目录下的工作牌识别。")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH, help="待检测图片路径")
    parser.add_argument("--image-dir", type=Path, help="批量检测图片目录")
    parser.add_argument(
        "--camera",
        nargs="?",
        const="csi",
        help="实时识别摄像头来源；不带值时默认使用 csi，可选 csi、usb 或数字序号，例如 0",
    )
    parser.add_argument(
        "--person-model",
        type=Path,
        default=DEFAULT_PERSON_MODEL_PATH,
        help="人员检测模型路径，默认使用项目根目录下的 yolov8n.pt",
    )
    parser.add_argument(
        "--zero-shot-model",
        default=DEFAULT_ZERO_SHOT_MODEL_NAME,
        help="本地零样本检测模型目录，默认使用 work_badge/models/grounding-dino-tiny",
    )
    parser.add_argument(
        "--zero-shot-repo-id",
        default=DEFAULT_ZERO_SHOT_MODEL_REPO_ID,
        help="下载本地零样本模型时使用的 Hugging Face 仓库 ID",
    )
    parser.add_argument(
        "--download-zero-shot-model",
        action="store_true",
        help="如果本地零样本模型不存在，则下载到本地模型目录",
    )
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="人员检测置信度阈值")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="人员检测 NMS IoU 阈值")
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=DEFAULT_BOX_THRESHOLD,
        help="零样本工作牌检测框阈值",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=DEFAULT_TEXT_THRESHOLD,
        help="零样本文本匹配阈值",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=list(FEATURE_CONFIG.badge_queries),
        help="用于零样本检测的工作牌提示词，可传多个",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR, help="结果输出目录")
    parser.add_argument("--no-save-vis", action="store_true", help="不保存带框可视化结果")
    parser.add_argument("--frame-log-interval", type=int, default=10, help="每隔多少帧打印一次检测结果")
    parser.add_argument("--width", type=int, default=1280, help="CSI 摄像头宽度")
    parser.add_argument("--height", type=int, default=720, help="CSI 摄像头高度")
    parser.add_argument("--framerate", type=int, default=30, help="CSI 摄像头帧率")
    parser.add_argument("--flip-method", type=int, default=0, help="CSI 摄像头翻转方式")
    parser.add_argument("--warmup-frames", type=int, default=5, help="摄像头预热读取帧数")
    parser.add_argument("--max-failed-reads", type=int, default=30, help="连续读帧失败多少次后终止")
    parser.add_argument("--process-every-n-frames", type=int, default=5, help="跳帧推理：每隔 N 帧推理一次，减轻 Jetson 功耗压力，防止 over-current")
    parser.add_argument("--window-name", default="Work Badge Detection", help="实时识别窗口标题")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="只准备本地零样本模型，不执行识别",
    )
    return parser


def _badge_to_dict(detection) -> dict:
    return {
        "label": detection.label,
        "confidence": round(detection.confidence, 4),
        "xyxy": [round(value, 2) for value in detection.xyxy],
    }


def _person_to_dict(detection) -> dict:
    return {
        "label": detection.label,
        "confidence": round(detection.confidence, 4),
        "person_xyxy": [round(value, 2) for value in detection.person_xyxy],
        "chest_xyxy": [round(value, 2) for value in detection.chest_xyxy],
        "status": detection.status,
        "is_wearing_badge": detection.is_wearing_badge,
        "badge_detections": [_badge_to_dict(item) for item in detection.badge_detections],
    }


def _prediction_to_dict(prediction, save_vis: bool, output_dir: Path) -> dict:
    persons = [_person_to_dict(detection) for detection in prediction.detections]
    wearing_badge_persons = [item for item in persons if item["is_wearing_badge"]]
    return {
        "feature": FEATURE_CONFIG.name,
        "display_name": FEATURE_CONFIG.display_name,
        "image": str(prediction.image_path),
        "zero_shot_model": FEATURE_CONFIG.default_zero_shot_model,
        "queries": list(FEATURE_CONFIG.badge_queries),
        "detections": persons,
        "wearing_badge_persons": wearing_badge_persons,
        "detected": bool(wearing_badge_persons),
        "save_vis": save_vis,
        "output_dir": str(output_dir),
    }


def main() -> None:
    args = build_parser().parse_args()
    # 检查并修复人员检测模型路径
    person_model_path = Path(args.person_model)
    if not person_model_path.exists():
        # 如果默认上一级的路径不存在，尝试在当前目录和上一级目录中查找常用的 yolo 模型
        fallback_paths = [
            WORK_BADGE_DIR.parent / "yolov8n.pt",
            WORK_BADGE_DIR / "yolov8n.pt",
            WORK_BADGE_DIR.parent / "yolo11n.pt",
            WORK_BADGE_DIR / "yolo11n.pt"
        ]
        found = False
        for fallback in fallback_paths:
            if fallback.exists():
                person_model_path = fallback
                found = True
                print(f"提示: 未在 {args.person_model} 找到模型，自动回退使用 {fallback}")
                break
        
        if not found:
            raise FileNotFoundError(f"人员检测权重不存在: {args.person_model}。请确保模型已放置在当前或上一级目录，或使用 --person-model 参数显式指定路径。")

    print(f"使用人员检测模型: {person_model_path}")

    zero_shot_model_path = Path(args.zero_shot_model).resolve()
    local_model_dir = ensure_local_zero_shot_model(
        model_path=zero_shot_model_path,
        repo_id=args.zero_shot_repo_id,
        allow_download=args.download_zero_shot_model,
    )

    if args.prepare_only:
        print(
            json.dumps(
                {
                    "prepared": True,
                    "zero_shot_model": str(local_model_dir),
                    "repo_id": args.zero_shot_repo_id,
                    "default_zero_shot_model_path": str(DEFAULT_ZERO_SHOT_MODEL_PATH),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    detector = WorkBadgeDetector(
        person_model_name=str(person_model_path),
        zero_shot_model_name=str(local_model_dir),
        zero_shot_repo_id=args.zero_shot_repo_id,
        conf=args.conf,
        iou=args.iou,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        badge_queries=args.queries,
        allow_download=False,
    )

    if args.camera is not None:
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
            process_every_n_frames=args.process_every_n_frames,
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
            item = _prediction_to_dict(prediction, save_vis=not args.no_save_vis, output_dir=output_dir)
            item["zero_shot_model"] = str(local_model_dir)
            item["zero_shot_repo_id"] = args.zero_shot_repo_id
            item["queries"] = list(args.queries)
            results.append(item)
        result = {
            "items": results,
            "summary": summarize_predictions(predictions),
        }
    else:
        image_path = args.image.resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"测试图片不存在: {image_path}")
        prediction = detector.predict_image(image_path)
        if not args.no_save_vis:
            save_prediction_visualization(prediction, output_dir=output_dir, name="predict")
        result = _prediction_to_dict(prediction, save_vis=not args.no_save_vis, output_dir=output_dir)
        result["zero_shot_model"] = str(local_model_dir)
        result["zero_shot_repo_id"] = args.zero_shot_repo_id
        result["queries"] = list(args.queries)

    result_file = output_dir / "result.json"
    result["result_file"] = str(result_file)
    result_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

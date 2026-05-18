from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_registry import FEATURE_CONFIGS
from feature_registry import get_feature_config
from framework.detector import GenericDetector
from framework.detector import filter_detections
from framework.detector import summarize_predictions
from framework.runtime import PROJECT_ROOT


def build_parser(
    default_feature: str | None = None,
    fixed_feature: str | None = None,
) -> argparse.ArgumentParser:
    description = "测试目标识别模型的有效性。"
    parser = argparse.ArgumentParser(description=description)

    if fixed_feature is None:
        parser.add_argument(
            "--feature",
            default=default_feature or "helmet",
            choices=sorted(FEATURE_CONFIGS),
            help="功能标识，对应 test_images/models/outputs 下的子目录",
        )

    parser.add_argument("--model", help="模型路径，默认读取 models/<feature>/best.pt")
    parser.add_argument("--conf", type=float, help="置信度阈值")
    parser.add_argument("--iou", type=float, help="NMS IoU 阈值")
    parser.add_argument(
        "--positive-dir",
        type=Path,
        help="应当检测到目标的正样本目录，默认读取 test_images/<feature>/positive",
    )
    parser.add_argument(
        "--negative-dir",
        type=Path,
        help="不应当检测到目标的负样本目录，默认读取 test_images/<feature>/negative",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="预测结果输出目录，默认读取 outputs/<feature>",
    )
    parser.add_argument(
        "--expected-labels",
        nargs="*",
        help="只统计这些类别，默认读取该功能的配置标签",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="保存预测可视化结果",
    )
    parser.add_argument(
        "--min-positive-hit-rate",
        type=float,
        help="正样本最低命中率阈值，默认读取功能配置",
    )
    parser.add_argument(
        "--max-negative-false-rate",
        type=float,
        help="负样本最高误检率阈值，默认读取功能配置",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="可选，YOLO 数据集 yaml 路径；提供后会执行标准 val 测试",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="标准 val 测试的数据集划分，默认 test",
    )
    return parser


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_paths(args: argparse.Namespace, feature_name: str) -> dict[str, Path]:
    config = get_feature_config(feature_name)
    return {
        "model": Path(args.model) if args.model else config.default_model_path(PROJECT_ROOT),
        "positive_dir": args.positive_dir or (config.test_images_dir(PROJECT_ROOT) / "positive"),
        "negative_dir": args.negative_dir or (config.test_images_dir(PROJECT_ROOT) / "negative"),
        "output_dir": args.output_dir or config.outputs_dir(PROJECT_ROOT),
    }


def evaluate_directory(
    detector: GenericDetector,
    image_dir: Path,
    expected_labels: list[str],
    save_vis: bool,
    output_dir: Path,
    run_name: str,
):
    if not image_dir.exists():
        return [], {
            "total_images": 0,
            "detected_images": 0,
            "undetected_images": 0,
            "total_detections": 0,
            "avg_detections_per_image": 0.0,
        }

    predictions = detector.predict_directory(
        image_dir=image_dir,
        save=save_vis,
        project=output_dir if save_vis else None,
        name=run_name,
    )
    predictions = filter_detections(predictions, expected_labels=expected_labels)
    summary = summarize_predictions(predictions)
    return predictions, summary


def print_directory_summary(title: str, summary: dict[str, int | float]) -> None:
    print(f"\n[{title}]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def quick_effectiveness_test(
    args: argparse.Namespace,
    feature_name: str,
) -> dict[str, object]:
    config = get_feature_config(feature_name)
    paths = resolve_paths(args, feature_name)
    ensure_output_dir(paths["output_dir"])

    detector = GenericDetector(
        feature_config=config,
        model_name=str(paths["model"]),
        conf=args.conf,
        iou=args.iou,
    )

    expected_labels = (
        args.expected_labels if args.expected_labels is not None else list(config.expected_labels)
    )
    min_positive_hit_rate = (
        args.min_positive_hit_rate
        if args.min_positive_hit_rate is not None
        else config.min_positive_hit_rate
    )
    max_negative_false_rate = (
        args.max_negative_false_rate
        if args.max_negative_false_rate is not None
        else config.max_negative_false_rate
    )

    positive_predictions, positive_summary = evaluate_directory(
        detector=detector,
        image_dir=paths["positive_dir"],
        expected_labels=expected_labels,
        save_vis=args.save_vis,
        output_dir=paths["output_dir"],
        run_name="positive_predict",
    )
    negative_predictions, negative_summary = evaluate_directory(
        detector=detector,
        image_dir=paths["negative_dir"],
        expected_labels=expected_labels,
        save_vis=args.save_vis,
        output_dir=paths["output_dir"],
        run_name="negative_predict",
    )

    positive_total = len(positive_predictions)
    negative_total = len(negative_predictions)
    positive_hit_rate = (
        positive_summary["detected_images"] / positive_total if positive_total else 0.0
    )
    negative_false_rate = (
        negative_summary["detected_images"] / negative_total if negative_total else 0.0
    )
    effective = (
        positive_total > 0
        and positive_hit_rate >= min_positive_hit_rate
        and negative_false_rate <= max_negative_false_rate
    )

    report = {
        "feature": feature_name,
        "display_name": config.display_name,
        "model": str(paths["model"]),
        "expected_labels": expected_labels,
        "positive_dir": str(paths["positive_dir"]),
        "negative_dir": str(paths["negative_dir"]),
        "output_dir": str(paths["output_dir"]),
        "positive_summary": positive_summary,
        "negative_summary": negative_summary,
        "positive_hit_rate": round(positive_hit_rate, 4),
        "negative_false_rate": round(negative_false_rate, 4),
        "min_positive_hit_rate": min_positive_hit_rate,
        "max_negative_false_rate": max_negative_false_rate,
        "effective": effective,
    }

    print("\n[功能信息]")
    print(
        json.dumps(
            {"feature": feature_name, "display_name": config.display_name},
            ensure_ascii=False,
            indent=2,
        )
    )
    print("\n[模型类别]")
    print(json.dumps(detector.class_names, ensure_ascii=False, indent=2))
    print_directory_summary("正样本统计", positive_summary)
    print_directory_summary("负样本统计", negative_summary)
    print("\n[有效性结论]")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if positive_total == 0:
        print("\n提示：正样本目录为空，无法判断模型是否有效。")
    elif not negative_predictions:
        print("\n提示：负样本目录为空，只能看命中率，无法衡量误检率。")

    return report


def standard_validation_test(
    args: argparse.Namespace,
    feature_name: str,
) -> None:
    config = get_feature_config(feature_name)
    paths = resolve_paths(args, feature_name)
    detector = GenericDetector(
        feature_config=config,
        model_name=str(paths["model"]),
        conf=args.conf,
        iou=args.iou,
    )
    metrics = detector.validate(data=args.data, split=args.split)

    result = {
        "feature": feature_name,
        "model": str(paths["model"]),
        "data": str(args.data),
        "split": args.split,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    print("\n[标准评估结果]")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main(
    default_feature: str | None = None,
    fixed_feature: str | None = None,
) -> None:
    parser = build_parser(default_feature=default_feature, fixed_feature=fixed_feature)
    args = parser.parse_args()

    feature_name = fixed_feature or args.feature
    if args.data:
        standard_validation_test(args, feature_name)
        return

    quick_effectiveness_test(args, feature_name)

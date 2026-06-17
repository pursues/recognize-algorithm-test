from __future__ import annotations

import argparse
import json
import random
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
from camera import open_camera, open_hk_cameras
from detector import PersonDetector
from person import PersonIntrusionProcessor, get_intrusion_zone

# 行为处理器路由表：行为名 → 处理器模块
BEHAVIOR_HANDLERS: dict[str, object] = {}
_behaviors = {
    "fallDown": "fallDown",
    "fireSmoke": "fireSmoke",
    "helmet": "helmet",
    "phone": "phone",
    "smoking": "smoking",
    "faceMask": "faceMask",
}
for _behavior_name, _folder_name in _behaviors.items():
    try:
        _mod = __import__(_folder_name, fromlist=["process"])
        BEHAVIOR_HANDLERS[_behavior_name] = _mod.process
    except ImportError as e:
        print(f"警告: 无法加载行为处理器 '{_behavior_name}' ({_folder_name}/): {e}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="统一推理入口 - 使用 best.pt 多类别模型，识别后路由到各行为处理器"
    )
    # 图片模式
    parser.add_argument("--image", type=Path, help="单张图片路径，不提供则扫描 imgs 目录全部图片")
    parser.add_argument("--random", action="store_true", help="从 imgs 目录随机选一张图片测试")
    # 摄像头模式
    parser.add_argument("--camera", help="摄像头来源。可选: csi、usb、hk、multi-hk 或数字序号(如 0)")
    # 模型参数
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="模型路径")
    parser.add_argument("--conf", type=float, help="置信度阈值")
    parser.add_argument("--iou", type=float, help="NMS IoU 阈值")
    # 输出
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR, help="输出根目录")
    parser.add_argument("--no-save-vis", action="store_true", help="不保存可视化结果")
    # 摄像头参数
    parser.add_argument("--frame-log-interval", type=int, default=30, help="每隔多少帧打印一次检测结果")
    parser.add_argument("--width", type=int, default=1280, help="摄像头宽度")
    parser.add_argument("--height", type=int, default=720, help="摄像头高度")
    parser.add_argument("--framerate", type=int, default=30, help="摄像头帧率")
    parser.add_argument("--flip-method", type=int, default=0, help="摄像头翻转方式")
    parser.add_argument("--warmup-frames", type=int, default=5, help="摄像头预热读取帧数")
    parser.add_argument("--max-failed-reads", type=int, default=30, help="连续读帧失败多少次后终止")
    parser.add_argument("--window-name", default="Behavior Detection", help="窗口标题")
    parser.add_argument("--skip-frames", type=int, default=30, help="检测跳帧间隔")
    return parser


# =============================================================================
# 行为路由核心函数
# =============================================================================

def route_behaviors(detections: list, frame, source: str, output_root: Path):
    """将检测结果按行为分类，路由到各处理器的 process() 函数。

    参数:
        detections: 模型检测结果列表
        frame: 当前帧/图片
        source: 来源标识
        output_root: 输出根目录
    """
    behaviors = classify_behaviors(detections)

    if not behaviors:
        return

    for behavior_name, matched_dets in behaviors.items():
        handler = BEHAVIOR_HANDLERS.get(behavior_name)
        if handler is None:
            info = BEHAVIOR_CONFIG.get(behavior_name, {})
            print(f"[路由] {info.get('display_name', behavior_name)}: 处理器未加载")
            continue

        info = BEHAVIOR_CONFIG.get(behavior_name, {})
        output_dir = output_root / behavior_name
        try:
            result = handler(
                detections=matched_dets,
                frame=frame,
                source=source,
                output_dir=output_dir,
            )
            labels = set(d.label for d in matched_dets)
            max_conf = max(d.confidence for d in matched_dets)
            print(f"[路由] → {info.get('display_name', behavior_name)}: "
                  f"标签={labels} 最高置信度={max_conf:.2f}")
        except Exception as e:
            print(f"[路由] {behavior_name} 处理失败: {e}")


# =============================================================================
# 摄像头模式
# =============================================================================

def run_multi_camera_loop(args) -> None:
    """多路摄像头实时推理主循环。"""
    initialize_runtime()

    detector = PersonDetector(
        model_name=str(args.model.resolve()),
        conf=args.conf,
        iou=args.iou,
    )
    print(f"模型已加载: {args.model}, 类别数={len(detector.class_names)}")

    cameras = open_hk_cameras(width=args.width, height=args.height)
    print(f"多路实时检测已启动，共 {len(cameras)} 路摄像头，按 Q 退出")

    camera_processors: dict[str, PersonIntrusionProcessor] = {}
    camera_frame_counts: dict[str, int] = {}
    camera_failed_reads: dict[str, int] = {}
    camera_last_predictions: dict[str, object] = {}
    output_root = args.output_dir.resolve()

    try:
        for _ in range(max(args.warmup_frames, 0)):
            for cam_name, cap in cameras:
                cap.read()

        while True:
            for cam_name, cap in cameras:
                if cam_name not in camera_frame_counts:
                    camera_frame_counts[cam_name] = 0
                    camera_failed_reads[cam_name] = 0
                    camera_processors[cam_name] = PersonIntrusionProcessor()

                processor = camera_processors[cam_name]

                try:
                    ret, frame = cap.read()
                except Exception:
                    camera_failed_reads[cam_name] += 1
                    if camera_failed_reads[cam_name] >= args.max_failed_reads:
                        print(f"[{cam_name}] 连续 {camera_failed_reads[cam_name]} 次读帧失败，跳过")
                    continue

                if not ret:
                    camera_failed_reads[cam_name] += 1
                    if camera_failed_reads[cam_name] >= args.max_failed_reads:
                        print(f"[{cam_name}] 连续 {camera_failed_reads[cam_name]} 次读帧失败，跳过")
                    continue
                camera_failed_reads[cam_name] = 0

                if frame.shape[1] != args.width or frame.shape[0] != args.height:
                    frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)

                camera_frame_counts[cam_name] += 1
                frame_count = camera_frame_counts[cam_name]
                should_detect = frame_count % args.skip_frames == 0

                results = detector.track_or_detect(frame, conf=args.conf, iou=args.iou)
                prediction = detector.build_prediction(results[0], image_path=Path(f"camera:{cam_name}"))
                camera_last_predictions[cam_name] = prediction

                if should_detect and prediction.detected:
                    # 人员闯入检测
                    person_dets = [d for d in prediction.detections
                                   if d.label.lower() in ("person",)]
                    if person_dets:
                        processor.process_frame(
                            detections=prediction.detections,
                            frame=frame,
                            frame_count=frame_count,
                            camera=cam_name,
                        )
                    # 行为路由
                    route_behaviors(
                        prediction.detections, frame,
                        source=cam_name, output_root=output_root,
                    )
                elif should_detect and processor.is_tracking_mode:
                    processor.reset()
                    print(f"[{cam_name}] 无人员，退出跟踪模式")

                annotated = detector.draw_detections(frame, prediction)
                h_f, w_f = annotated.shape[:2]
                zx1, zy1, zx2, zy2 = get_intrusion_zone(w_f, h_f)
                cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), (0, 0, 255), 2)
                cv2.putText(annotated, f"{cam_name} INTRUSION ZONE", (zx1, max(zy1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                if args.frame_log_interval > 0 and frame_count % args.frame_log_interval == 0:
                    mode_text = "闯入跟踪模式" if processor.is_tracking_mode else "检测模式"
                    print(f"[{cam_name}] 帧#{frame_count} {mode_text}")

                cv2.imshow(cam_name, annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for _, cap in cameras:
            cap.release()
        cv2.destroyAllWindows()


def run_camera_loop(args) -> None:
    """单路摄像头实时推理主循环。"""
    initialize_runtime()

    detector = PersonDetector(
        model_name=str(args.model.resolve()),
        conf=args.conf,
        iou=args.iou,
    )
    print(f"模型已加载: {args.model}, 类别数={len(detector.class_names)}")

    cap = open_camera(
        camera=args.camera, width=args.width, height=args.height,
        framerate=args.framerate, flip_method=args.flip_method,
    )
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {args.camera}")

    print(f"实时检测已启动，摄像头={args.camera}，按 Q 退出")

    output_root = args.output_dir.resolve()
    processors: dict[str, PersonIntrusionProcessor] = {}
    frame_count = 0
    failed_reads = 0
    last_prediction = None

    try:
        for _ in range(max(args.warmup_frames, 0)):
            cap.read()

        while True:
            try:
                ret, frame = cap.read()
            except Exception as exc:
                raise RuntimeError(f"读取摄像头帧失败: camera={args.camera}") from exc

            if not ret:
                failed_reads += 1
                if failed_reads >= args.max_failed_reads:
                    raise RuntimeError(f"连续 {failed_reads} 次读帧失败")
                continue
            failed_reads = 0

            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)

            frame_count += 1
            should_detect = frame_count % args.skip_frames == 0

            results = detector.track_or_detect(frame, conf=args.conf, iou=args.iou)
            prediction = detector.build_prediction(results[0], image_path=Path(f"camera:{args.camera}"))
            last_prediction = prediction

            if should_detect and prediction.detected:
                # 人员闯入
                person_dets = [d for d in prediction.detections
                               if d.label.lower() in ("person",)]
                if person_dets:
                    if "person" not in processors:
                        processors["person"] = PersonIntrusionProcessor()
                    processors["person"].process_frame(
                        detections=prediction.detections,
                        frame=frame, frame_count=frame_count, camera=args.camera,
                    )
                # 行为路由
                route_behaviors(
                    prediction.detections, frame,
                    source=f"camera:{args.camera}", output_root=output_root,
                )
            elif should_detect:
                if "person" in processors and processors["person"].is_tracking_mode:
                    processors["person"].reset()
                    print("[DETECT] 无人员，退出跟踪模式")

            annotated = detector.draw_detections(frame, last_prediction)
            h_f, w_f = annotated.shape[:2]
            zx1, zy1, zx2, zy2 = get_intrusion_zone(w_f, h_f)
            cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), (0, 0, 255), 2)
            cv2.putText(annotated, "INTRUSION ZONE", (zx1, max(zy1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if args.frame_log_interval > 0 and frame_count % args.frame_log_interval == 0:
                mode_text = ""
                if "person" in processors and processors["person"].is_tracking_mode:
                    mode_text = f"闯入跟踪模式(已报警{processors['person'].alerted_count}个目标)"
                else:
                    mode_text = "检测模式"
                print(f"[DETECT] 帧#{frame_count} {mode_text}")

            cv2.imshow(args.window_name, annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


# =============================================================================
# 图片模式
# =============================================================================

def _collect_image_paths(specified_image: Path | None, random_pick: bool) -> list[Path]:
    """收集 imgs/ 下的所有图片路径。"""
    if specified_image:
        return [specified_image.resolve()]

    candidates = []
    if TEST_IMAGES_DIR.exists():
        for ext in IMAGE_SUFFIXES:
            candidates.extend(sorted(TEST_IMAGES_DIR.glob(f"*{ext}")))

    if not candidates:
        return []

    if random_pick:
        return [random.choice(candidates)]
    return candidates


def run_image_mode(args) -> None:
    """图片检测模式 - 推理后路由到对应行为处理器。"""
    initialize_runtime()
    model_path = args.model.resolve()

    detector = PersonDetector(
        model_name=str(model_path),
        conf=args.conf,
        iou=args.iou,
    )
    print(f"模型已加载: {model_path}")
    print(f"模型类别: {detector.class_names}")

    image_paths = _collect_image_paths(args.image, args.random)
    if not image_paths:
        raise FileNotFoundError(f"未找到测试图片。请使用 --image 指定路径，或在 {TEST_IMAGES_DIR} 下放置图片。")

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    save_vis = not args.no_save_vis

    all_results = []
    for image_path in image_paths:
        prediction, _ = detector.predict_image(
            image_path=image_path,
            save=save_vis,
            output_dir=output_root if save_vis else None,
        )

        # 分离人员检测和行为检测
        person_dets = [d for d in prediction.detections
                       if d.label.lower() in ("person",)]
        behavior_dets = [d for d in prediction.detections
                         if d.label.lower() not in ("person",)]

        # 按行为分类
        behaviors = classify_behaviors(prediction.detections)
        detected_labels = list(set(d.label for d in prediction.detections))

        # 路由到各行为处理器
        if behaviors:
            frame_for_handler = cv2.imread(str(image_path))
            route_behaviors(
                prediction.detections, frame_for_handler,
                source=str(image_path), output_root=output_root,
            )

        # ---- 打印清晰的结果 ----
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
        all_results.append(result)

    # 汇总
    total = len(all_results)
    detected_count = sum(1 for r in all_results if r["has_detection"])
    with_behavior = sum(1 for r in all_results if r["matched_behaviors"])
    print(f"\n汇总: 总数={total} | 有检测={detected_count} | 匹配行为={with_behavior}")

    result_file = output_root / "result.json"
    result_file.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"详细结果已保存至 {result_file}")


# =============================================================================
# 入口
# =============================================================================

def main() -> None:
    args = build_parser().parse_args()

    if args.camera == "multi-hk":
        run_multi_camera_loop(args)
    elif args.camera:
        run_camera_loop(args)
    else:
        run_image_mode(args)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import (
    DEFAULT_MODEL_PATH,
    OUTPUTS_DIR,
    TEST_IMAGES_DIR,
    cv2,
    initialize_runtime,
)
from camera import open_camera, open_hk_cameras
from detector import PersonDetector
from person import FEATURE_CONFIG, PersonIntrusionProcessor, get_intrusion_zone


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="模型检测主入口 - 识别不同行为后走不同处理流程。")
    parser.add_argument("--image", type=Path, help="待检测图片路径。如果不提供，将尝试测试 imgs 目录下的图片。")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="模型路径")
    parser.add_argument(
        "--camera",
        help="实时识别摄像头来源。可选: csi、usb、hk、multi-hk 或数字序号(如 0)",
    )
    parser.add_argument("--conf", type=float, help="置信度阈值")
    parser.add_argument("--iou", type=float, help="NMS IoU 阈值")
    parser.add_argument("--frame-log-interval", type=int, default=30, help="每隔多少帧打印一次检测结果")
    parser.add_argument("--width", type=int, default=1280, help="摄像头宽度")
    parser.add_argument("--height", type=int, default=720, help="摄像头高度")
    parser.add_argument("--framerate", type=int, default=30, help="摄像头帧率")
    parser.add_argument("--flip-method", type=int, default=0, help="摄像头翻转方式")
    parser.add_argument("--warmup-frames", type=int, default=5, help="摄像头预热读取帧数")
    parser.add_argument("--max-failed-reads", type=int, default=30, help="连续读帧失败多少次后终止")
    parser.add_argument("--window-name", default="Person Detection", help="实时识别窗口标题")
    parser.add_argument("--skip-frames", type=int, default=30, help="检测跳帧间隔")
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


def run_multi_camera_loop(args) -> None:
    """多路摄像头实时推理主循环。

    同时处理多路 RTSP 流，每路独立维护处理器状态。
    所有路共享同一个检测模型，按轮询方式逐路推理。
    """
    initialize_runtime()

    detector = PersonDetector(
        model_name=str(args.model.resolve()),
        conf=args.conf,
        iou=args.iou,
    )

    cameras = open_hk_cameras(
        width=args.width,
        height=args.height,
    )

    print(f"多路实时检测已启动，共 {len(cameras)} 路摄像头，按 Q 退出")

    # 每路摄像头独立的状态
    camera_processors: dict[str, PersonIntrusionProcessor] = {}
    camera_frame_counts: dict[str, int] = {}
    camera_failed_reads: dict[str, int] = {}
    camera_last_predictions: dict[str, object] = {}

    try:
        # 预热
        for _ in range(max(args.warmup_frames, 0)):
            for cam_name, cap in cameras:
                cap.read()

        while True:
            for cam_name, cap in cameras:
                # 初始化状态
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

                # 推理
                results = detector.track_or_detect(frame, conf=args.conf, iou=args.iou)
                prediction = detector.build_prediction(results[0], image_path=Path(f"camera:{cam_name}"))
                camera_last_predictions[cam_name] = prediction

                if should_detect and prediction.detected:
                    processor.process_frame(
                        detections=prediction.detections,
                        frame=frame,
                        frame_count=frame_count,
                        camera=cam_name,
                    )
                elif should_detect and processor.is_tracking_mode:
                    processor.reset()
                    print(f"[{cam_name}] 无人员，退出跟踪模式")

                # 绘制
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
    """实时摄像头主循环：取流 → 模型推理 → 分类路由 → 行为处理"""
    initialize_runtime()

    detector = PersonDetector(
        model_name=str(args.model.resolve()),
        conf=args.conf,
        iou=args.iou,
    )

    cap = open_camera(
        camera=args.camera,
        width=args.width,
        height=args.height,
        framerate=args.framerate,
        flip_method=args.flip_method,
    )
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {args.camera}")

    print(f"实时检测已启动，摄像头={args.camera}，按 Q 退出")

    processors: dict[str, PersonIntrusionProcessor] = {}
    frame_count = 0
    failed_reads = 0
    # 缓存最近一次检测结果，中间帧复用，保证画面不闪烁
    last_prediction = None

    try:
        for _ in range(max(args.warmup_frames, 0)):
            cap.read()

        while True:
            try:
                ret, frame = cap.read()
            except Exception as exc:
                raise RuntimeError(
                    f"读取摄像头帧失败: camera={args.camera}"
                ) from exc

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

            # 始终运行推理来保持 ByteTrack 状态连续、画面一致
            results = detector.track_or_detect(frame, conf=args.conf, iou=args.iou)
            prediction = detector.build_prediction(results[0], image_path=Path(f"camera:{args.camera}"))
            last_prediction = prediction

            if should_detect and prediction.detected:
                # 检测帧 + 有人 → 走 person 状态机
                if "person" not in processors:
                    processors["person"] = PersonIntrusionProcessor()
                processors["person"].process_frame(
                    detections=prediction.detections,
                    frame=frame,
                    frame_count=frame_count,
                    camera=args.camera,
                )
            elif should_detect:
                # 检测帧 + 无人 → 重置
                if "person" in processors and processors["person"].is_tracking_mode:
                    processors["person"].reset()
                    print("[DETECT] 无人员，退出跟踪模式")

            # 始终在最终帧上绘制检测框 + 红线框（每帧一致，不闪烁）
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


def run_image_mode(args) -> None:
    """图片检测模式"""
    initialize_runtime()
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
                    "label": d.label,
                    "confidence": round(d.confidence, 4),
                    "xyxy": [round(v, 2) for v in d.xyxy],
                    "track_id": d.track_id,
                }
                for d in prediction.detections
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

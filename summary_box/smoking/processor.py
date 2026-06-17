"""抽烟检测处理逻辑。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

DISPLAY_NAME = "抽烟检测"
EXPECTED_LABELS = ()


def process(
    detections: list[Any],
    frame: Any | None = None,
    source: str = "",
    output_dir: Path | None = None,
) -> dict:
    """处理抽烟检测结果。"""
    labels = [d.label for d in detections]
    confs = [d.confidence for d in detections]

    print(f"\n{'='*60}")
    print(f"[smoking/processor.py] 接收到模型推理结果 - {DISPLAY_NAME}")
    print(f"{'='*60}")
    print(f"  来源: {source}")
    print(f"  检测目标数: {len(detections)}")
    print(f"{'─'*60}")

    for i, det in enumerate(detections, 1):
        x1, y1, x2, y2 = [round(v, 2) for v in det.xyxy]
        tid = det.track_id if det.track_id is not None else "N/A"
        print(f"  [{i}] 标签: {det.label:<16s} 置信度: {det.confidence:.4f}  "
              f"坐标: ({x1}, {y1}, {x2}, {y2})  track_id: {tid}")

    print(f"{'─'*60}")

    label_counts: dict[str, int] = {}
    for lb in labels:
        label_counts[lb] = label_counts.get(lb, 0) + 1
    label_summary = "  ".join(f"{lb}x{c}" for lb, c in label_counts.items())
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    max_conf = max(confs) if confs else 0.0
    min_conf = min(confs) if confs else 0.0

    print(f"  标签统计: {label_summary}")
    print(f"  置信度: 最高={max_conf:.4f}  最低={min_conf:.4f}  平均={avg_conf:.4f}")

    saved_path = ""
    if frame is not None and output_dir is not None:
        import cv2
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det.xyxy)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, f"{det.label} {det.confidence:.2f}",
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"smoking_{timestamp}.jpg"
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_path = str(output_dir / save_name)
        cv2.imwrite(saved_path, annotated)
        print(f"  已保存可视化: {saved_path}")

    print(f"{'='*60}")

    return {
        "behavior": "smoking",
        "display_name": DISPLAY_NAME,
        "detected_labels": list(set(labels)),
        "max_confidence": round(max_conf, 4),
        "min_confidence": round(min_conf, 4),
        "avg_confidence": round(avg_conf, 4),
        "label_counts": label_counts,
        "source": source,
        "saved_path": saved_path,
    }

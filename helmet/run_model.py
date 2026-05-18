import cv2
from ultralytics import YOLO

# Configuration
MODEL_PATH = 'helmet5_fixed.pt'
CAMERA = 'csi'          # 'csi' for IMX219, 'usb' for USB camera
CONFIDENCE = 0.25        # detection confidence threshold

print(f"Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print(f"Model loaded. Classes: {model.names}")

# Open camera
if CAMERA == 'csi':
    gst_str = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink"
    )
    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
else:
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    exit()

print("Real-time inference started. Press 'Q' to quit...")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1
    # Run inference
    results = model(frame, conf=CONFIDENCE, verbose=False)

    # Print detection results every 10 frames
    if frame_count % 10 == 0:
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = model.names[cls_id]
                    print(f"[DETECT] {name}: {conf:.2f}")
            else:
                print("[DETECT] No objects in this frame.")

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Show in window
    cv2.imshow('Helmet Detection', annotated_frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program stopped.")
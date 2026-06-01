import cv2
import time

# ================== 配置 ==================
RTSP_URL = "rtsp://admin:JIANGhd99110@192.168.5.113:554/Streaming/Channels/101"

# H.265 硬件解码 Pipeline（已验证可行）
gst_pipeline = (
    f"rtspsrc location={RTSP_URL} latency=200 protocols=tcp ! "
    "rtph265depay ! h265parse ! "
    "nvv4l2decoder ! "              # Jetson 硬件解码
    "nvvidconv ! "
    "video/x-raw,format=BGRx ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "
    "appsink drop=1 max-buffers=2"
)

print("正在连接海康 H.265 摄像头...")

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

# 重试机制
for i in range(5):
    if cap.isOpened():
        print("✅ 摄像头连接成功！开始行为检测...")
        break
    print(f"第 {i+1} 次尝试失败，正在重试...")
    cap.release()
    time.sleep(2)
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
else:
    print("❌ 连接失败")
    exit(1)

# ================== 主循环 ==================
frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        
        if not ret or frame is None:
            print("⚠️ 读取帧失败，正在重连...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            continue

        frame_count += 1

        # ================== 在这里放入你的大模型 ==================
        # 例如使用 YOLO：
        # results = model(frame)
        # annotated_frame = results[0].plot()
        
        # 显示画面（调试用，正式跑模型时可以注释掉这行）
        cv2.imshow("Hikvision H.265 - Behavior Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if frame_count % 30 == 0:
            print(f"已处理 {frame_count} 帧")

except KeyboardInterrupt:
    print("\n🛑 程序已停止")
finally:
    cap.release()
    cv2.destroyAllWindows()
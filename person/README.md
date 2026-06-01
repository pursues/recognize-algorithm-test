# 人员识别 (Person Detection)

基于 YOLOv8 目标检测算法实现的人员识别模块。

## 1. 目标

在监控画面、图片或摄像头实时流中识别"人员"。
- **目标检测**：使用轻量级的 `yolov8n.pt` 预训练模型，仅提取 `person` (人员) 类别。
- **目标跟踪**：集成了 [ByteTrack](https://github.com/FoundationVision/ByteTrack) 算法（通过 `ultralytics` 内置支持），在视频或摄像头流中能够稳定跟踪人员 ID。
- **警报触发**：只要检测到有人，就会自动调用警报接口（当前为模拟接口，后续可替换为真实接口）。

## 2. 目录结构

```text
person/
├── imgs/               # 存放测试用的人员图片
├── outputs/            # 存放识别后的可视化结果及 JSON 数据
├── .ultralytics/       # ultralytics 配置与缓存目录
├── person.py           # 核心逻辑：YOLOv8 检测、ByteTrack 跟踪、警报接口调用
├── person-test.py      # 命令行入口脚本：支持单图测试、批量测试及摄像头实时检测
├── yolov8n.pt          # YOLOv8 nano 预训练模型权重（首次运行自动下载）
└── README.md           # 本说明文件
```

## 3. 依赖安装

请确保已安装 `ultralytics` 以及 OpenCV（用于摄像头实时识别）。

```bash
pip install ultralytics opencv-python
```

## 4. 启动命令

### 测试图片识别

将测试图片放入 `imgs` 目录中，然后运行：

```bash
# 测试 imgs 目录下的所有图片
python person-test.py

# 测试单张指定的图片
python person-test.py --image imgs/test.jpg
```

### 摄像头实时识别

默认情况下，目标跟踪功能是**默认关闭**的，只使用单帧检测。

```bash
# 使用 USB 摄像头 (如电脑自带摄像头)
python person-test.py --camera usb

# 使用 CSI 摄像头 (Jetson 等边缘设备)
python person-test.py --camera csi

# 尝试开启目标跟踪（需环境中已安装 lap）
python person-test.py --camera csi --use-tracking
```

### 常用参数说明

- `--model`: 模型文件路径，默认使用 `yolov8n.pt`。
- `--conf`: 目标检测置信度阈值。
- `--use-tracking`: 尝试开启 ByteTrack 跟踪（若缺少依赖会自动降级）。
- `--camera`: 摄像头来源，支持 `usb`, `csi`, 或 `0`、`1` 等数字序号。

## 5. 警报接口说明

### 模拟警报接口

当前代码中使用的是模拟警报接口，定义在 [person.py](file:///Users/pursues/Desktop/project/huitian/python/recognize-algorithm-test/person/person.py) 中的 `send_alert` 函数。

当检测到人员时，会自动调用该接口并打印警报信息。

### 替换为真实警报接口

当你的真实警报接口准备好后，只需修改 [person.py](file:///Users/pursues/Desktop/project/huitian/python/recognize-algorithm-test/person/person.py) 中的 `send_alert` 函数即可：

```python
def send_alert(detection_count: int, image_path: str = "", frame_info: dict | None = None) -> dict:
    """
    真实警报接口调用示例
    
    参数:
        detection_count: 检测到的人员数量
        image_path: 触发警报的图片路径或来源
        frame_info: 额外的帧信息（用于实时识别时传递）
    
    返回:
        警报接口响应结果
    """
    # 替换为你的真实接口调用
    # 例如:
    # import requests
    # response = requests.post(
    #     "http://your-alert-api.com/api/alert",
    #     json={
    #         "alert_type": "person_detected",
    #         "detection_count": detection_count,
    #         "image_path": image_path,
    #         "frame_info": frame_info,
    #     }
    # )
    # return response.json()
    
    pass
```

该函数在以下两个地方被调用：
1. **图片识别**：[predict_image](file:///Users/pursues/Desktop/project/huitian/python/recognize-algorithm-test/person/person.py#L256-L260) 方法中，当检测到人员时调用
2. **实时识别**：[run_camera_inference](file:///Users/pursues/Desktop/project/huitian/python/recognize-algorithm-test/person/person.py#L323-L333) 方法中，带有冷却时间机制（默认 30 帧冷却），避免频繁触发警报

## 6. 关于 ByteTrack 依赖 (`lap` 库)

[ByteTrack](https://github.com/FoundationVision/ByteTrack) 是一种简单高效的多目标跟踪算法。虽然 `ultralytics` 内部集成了 ByteTrack 算法逻辑，但它底层依赖于 `lap` 库来进行线性分配计算（Linear Assignment）。
在 Jetson 等边缘设备上安装 `lap` 库需要 C++ 编译环境，为了达到"零额外下载/零编译"的目标，代码中已经做好了**自动降级**处理：
1. 默认情况下，摄像头推理不开启 `--use-tracking`，直接使用原生的 `predict`。
2. 即使手动加上 `--use-tracking`，如果在运行时捕获到 `ModuleNotFoundError: No module named 'lap'`，代码会自动打印警告并平滑降级到单帧检测，保证程序不会崩溃。

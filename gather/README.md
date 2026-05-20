# 人员聚集识别 (Gather Detection)

基于 YOLOv8 与 ByteTrack 目标跟踪算法实现的人员聚集识别模块。

## 1. 目标

在监控画面或图片中识别 "人员聚集" 现象。
- **目标检测**：使用轻量级的 `yolov8n.pt` 预训练模型，仅提取 `person` (人员) 类别。
- **目标跟踪**：集成了 [ByteTrack](https://github.com/FoundationVision/ByteTrack) 算法（通过 `ultralytics` 内置支持），在视频或摄像头流中能够稳定跟踪人员 ID，减少目标丢失。
- **聚集判定**：通过计算人员之间的像素距离，使用连通图（并查集）算法进行聚类。如果一个群体中的人数大于等于设定的阈值（默认 3 人），且他们之间的距离小于设定阈值（默认 150 像素），则判定为“人员聚集”。

## 2. 目录结构

```text
gather/
├── imgs/               # 存放测试用的人员聚集图片
├── outputs/            # 存放识别后的可视化结果及 JSON 数据
├── .ultralytics/       # ultralytics 配置与缓存目录
├── gather.py           # 核心逻辑：YOLOv8 检测、ByteTrack 跟踪、聚集距离计算
├── gather-test.py      # 命令行入口脚本：支持单图测试、批量测试及摄像头实时检测
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
python gather-test.py

# 测试单张指定的图片
python gather-test.py --image imgs/test.jpg

# 调整聚集判断参数 (距离阈值100，最少聚集人数4)
python gather-test.py --distance-threshold 100 --min-gather-count 4
```

### 摄像头实时识别

默认情况下，为保证在 Jetson 等边缘设备上的兼容性（避免编译 `lap` C++ 依赖），目标跟踪功能是**默认关闭**的，只使用单帧检测来判断聚集。

```bash
# 使用 USB 摄像头 (如电脑自带摄像头)
python gather-test.py --camera usb

# 使用 CSI 摄像头 (Jetson 等边缘设备)
python gather-test.py --camera csi

# 尝试开启目标跟踪（需环境中已安装 lap）
python gather-test.py --camera csi --use-tracking
```

### 常用参数说明

- `--model`: 模型文件路径，默认使用 `yolov8n.pt`。
- `--conf`: 目标检测置信度阈值。
- `--distance-threshold`: 判定两人属于同一群体的最大像素距离，默认 `150.0`。
- `--min-gather-count`: 判定为“聚集”的最少人数，默认 `3`。
- `--use-tracking`: 尝试开启 ByteTrack 跟踪（若缺少依赖会自动降级）。
- `--camera`: 摄像头来源，支持 `usb`, `csi`, 或 `0`、`1` 等数字序号。

## 5. 关于 ByteTrack 依赖 (`lap` 库)

[ByteTrack](https://github.com/FoundationVision/ByteTrack) 是一种简单高效的多目标跟踪算法。虽然 `ultralytics` 内部集成了 ByteTrack 算法逻辑，但它底层依赖于 `lap` 库来进行线性分配计算（Linear Assignment）。
在 Jetson 等边缘设备上安装 `lap` 库需要 C++ 编译环境，为了达到“零额外下载/零编译”的目标，代码中已经做好了**自动降级**处理：
1. 默认情况下，摄像头推理不开启 `--use-tracking`，直接使用原生的 `predict`。
2. 即使手动加上 `--use-tracking`，如果在运行时捕获到 `ModuleNotFoundError: No module named 'lap'`，代码会自动打印警告并平滑降级到单帧检测，保证程序不会崩溃。

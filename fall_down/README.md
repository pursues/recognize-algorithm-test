# 摔倒识别 (Fall Down Detection)

本项目提供了一个基于 YOLOv8 的摔倒识别模块，设计灵感来源于 `smoking` 模块，并且兼容 Edge Box 的摄像头实时识别。

## 目录结构

- `fall_down.py`: 摔倒识别核心逻辑，封装了 Ultralytics YOLOv8 模型，使用包围框宽高比（Aspect Ratio）以及目标类别判断是否摔倒。
- `fall_down_test.py`: 命令行测试入口，支持单张图片测试、目录批量测试以及摄像头实时识别。
- `imgs/`: 存放测试用的样本图片（如 `fall_down1.png`，`fall_down2.png`，`no_fall_down.png` 等）。
- `outputs/`: 脚本运行结果输出目录，包含识别后的可视化图片及 `result.json` 文件。
- `README.md`: 本项目说明文档。

## 设计目标与原理

通过调用现有的轻量级预训练目标检测模型（如 `yolov8n.pt`），检测画面中的 `person`（人）。
默认判定规则为：**检测到的人体边界框的宽高比 (Width / Height) > 1.0 时，判定为摔倒 (fall)**。

这个启发式规则非常轻量、处理速度快，可以直接在边缘设备中实时运行。如果对摔倒姿态识别有更复杂的诉求，也可以无缝切换为 Ultralytics 的姿态估计模型（如 `yolov8n-pose.pt`）。

## 依赖安装

请确保已安装 `ultralytics` 以及 OpenCV（用于摄像头实时识别）。
```bash
pip install ultralytics opencv-python
```

## 启动命令

### 1. 测试图片目录（默认测试 `imgs/` 目录）
自动读取 `imgs` 目录下的所有图片，并将画好检测框的结果输出到 `outputs/` 中：
```bash
python fall_down_test.py
```

### 2. 测试单张图片
```bash
python fall_down_test.py --image imgs/fall_down1.png
```

### 3. 实时摄像头识别
支持使用 USB 摄像头或 CSI 摄像头进行实时测试。

**使用 USB 摄像头（例如外接或内置摄像头）：**
```bash
python fall_down_test.py --camera usb
# 或指定摄像头序号
python fall_down_test.py --camera 0
```

**使用 CSI 摄像头（例如 Jetson 系列边缘设备）：**
```bash
python fall_down_test.py --camera csi
```

### 可选参数
- `--model`: 指定使用的 YOLOv8 模型权重，默认为 `yolov8n.pt`（首次运行会自动下载）。
- `--fall-threshold`: 摔倒判定阈值（边界框的 宽/高 比例），默认为 `1.0`。如果想更严格，可设为 `1.2` 或更高。
- `--conf`: 目标检测置信度阈值。
- `--no-save-vis`: 禁用保存可视化结果图片。

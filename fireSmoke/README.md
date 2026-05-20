# 火灾烟雾识别模块 (Fire and Smoke Detection)

本模块基于 YOLOv8 实现火灾和烟雾的目标检测，适用于图片测试及摄像头实时检测。

## 目录结构

```text
fireSmoke/
├── imgs/               # 测试图片存放目录
│   ├── fire.png        # 火灾测试图片
│   └── smoke.png       # 烟雾测试图片
├── outputs/            # 预测结果和可视化图像输出目录
│   ├── predict/        # 带有检测框的输出图片
│   └── result.json     # 检测结果的 JSON 格式数据
├── best.pt             # 预训练的 YOLOv8 火灾烟雾检测模型权重
├── fireSmoke.py        # 核心算法逻辑封装（模型加载、推理、摄像头读取等）
├── fireSmoke-test.py   # 命令行测试工具，用于单张图片或摄像头实时测试
└── README.md           # 本说明文档
```

## 目标与用途

该模块旨在提供一个开箱即用的火灾和烟雾检测工具，可集成到更复杂的监控或告警系统中。模型识别的目标标签包括：
- `Fire`（火灾）
- `smoke`（烟雾）

## 启动命令

确保当前所处虚拟环境已安装必要的依赖（`ultralytics`, `torch`, `opencv-python` 等）。

### 1. 图片检测测试

测试火灾图片：
```bash
python3 fireSmoke-test.py --image imgs/fire.png
```

测试烟雾图片：
```bash
python3 fireSmoke-test.py --image imgs/smoke.png
```

命令执行后，控制台会输出 JSON 格式的识别结果，并且会在 `outputs/predict` 目录下保存绘制了检测框的图片。

### 2. 摄像头实时检测

支持使用 USB 摄像头或 CSI 摄像头进行实时检测。按下键盘上的 `q` 键可退出实时检测窗口。

**使用默认 USB 摄像头（或传入数字序号 0）：**
```bash
python3 fireSmoke-test.py --camera usb
# 或者
python3 fireSmoke-test.py --camera 0
```

**使用 CSI 摄像头（适用于 Jetson 等边缘设备）：**
```bash
python3 fireSmoke-test.py --camera csi
```

### 其他参数
你可以通过 `-h` 参数查看更多配置选项，例如调整置信度阈值、摄像头分辨率等：
```bash
python3 fireSmoke-test.py -h
```

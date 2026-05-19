# play_phone

单独启用时，先将外层的 yolov8n.pt 复制到 play_phone 目录下。

```bash
cp /Users/pursues/Desktop/project/huitian/python/recognize-algorithm-test/yolov8n.pt play_phone/
```

## 目标

`play_phone` 用于识别“玩手机”行为，当前定义为：

- 画面中出现 `person`
- 该人附近或其人体框内出现 `cell phone`
- 结合手机框、人框、可选人脸框的位置关系，使用无训练启发式规则判断是否属于“手持手机并看手机”

当前方案不额外训练模型，直接复用开源预训练权重：

- 本地默认实现使用 `Ultralytics YOLO` 的 `yolov8n.pt`，其 COCO 预训练类别中包含 `person` 和 `cell phone`
- 如果你后续想尝试 Hugging Face 上的通用开源模型，可优先关注 `facebook/detr-resnet-50`、`IDEA-Research/grounding-dino-tiny`、`google/owlv2-base-patch16-ensemble`
- 截止当前检索结果，未发现公开可直接下载、专门针对“玩手机”单一行为且适合本项目直接离线落地的 GitHub Marketplace 开源权重；更常见的是通用检测模型或示例应用

## 目录结构

```text
play_phone/
├── imgs/                   # 测试图片
├── outputs/                # 识别输出目录，运行后自动生成
│   ├── predict/            # 带框可视化结果
│   └── result.json         # 检测结果汇总
├── play_phone.py           # 玩手机识别核心模块
├── play_phone_test.py      # 命令行测试入口
└── README.md               # 当前说明文档
```

## 启动命令

在项目根目录 `/Users/pursues/Desktop/project/huitian/python/recognize-algorithm-test` 下执行。

### 1. 批量检测 `play_phone/imgs` 下的全部图片

```bash
python3.11 play_phone/play_phone_test.py
```

### 2. 仅检测默认示例图

```bash
python3.11 play_phone/play_phone_test.py --default-image
```

### 3. 指定单张图片

```bash
python3.11 play_phone/play_phone_test.py --image play_phone/imgs/phone-game.jpg
```

### 4. 指定模型或调整阈值

```bash
python3.11 play_phone/play_phone_test.py --model yolov8s.pt --conf 0.2 --iou 0.45
```

### 5. USB 摄像头实时识别

```bash
python3.11 play_phone/play_phone_test.py --camera usb
```

### 6. CSI 摄像头实时识别

```bash
python3.11 play_phone/play_phone_test.py --camera csi
```

默认已经设置为每 `30` 帧跑一次模型推理。
非推理帧不会复用旧框，只会在画面左上角显示“最近一次检测结果”。

### 7. 指定数字摄像头序号

```bash
python3.11 play_phone/play_phone_test.py --camera 0
```

## 输出说明

- `outputs/result.json`：保存所有图片的检测明细、玩手机判定结果、命中的启发式规则
- `outputs/predict/*.jpg`：保存带框可视化图片
- 实时摄像头模式下会弹出窗口显示检测结果，按 `Q` 退出

## 判定说明

当前实现不是专门训练出来的“玩手机行为分类模型”，而是：

1. 使用开源预训练检测模型识别 `person` 和 `cell phone`
2. 检查手机是否与某个人体框关联
3. 结合手机是否位于人体上半身区域、是否接近人体中心、人脸与手机的相对位置等规则给出 `play_phone` 判定

因此它更适合做一个无需训练、可快速上线的基础版方案。如果后续需要更高精度，建议再引入专门的姿态估计、视线估计，或者专门标注“玩手机”数据集进行微调。

## 实时识别说明

- `--camera usb`：使用默认 USB 摄像头
- `--camera csi`：使用 Jetson 常见 CSI 摄像头管线
- `--camera 0`：使用数字序号打开摄像头
- `CSI` 默认已优化为 `640x480 @ 15fps`，如果你机器性能足够，再手动提高 `--width`、`--height`、`--framerate`
- 实时识别默认 `--infer-every-n-frames 30`，即每 30 帧跑一次模型
- 非推理帧不绘制旧检测框，只显示最近一次检测结果文字
- 可通过 `--frame-log-interval` 控制打印频率
- 可通过 `--window-name` 自定义窗口标题

例如你也可以显式指定：

```bash
python3.11 play_phone/play_phone_test.py --camera csi --infer-every-n-frames 30
```

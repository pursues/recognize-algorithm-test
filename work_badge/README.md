# work_badge

## 目标

`work_badge` 用于识别人员是否佩戴了工作牌，保持与 `work_clothes` 同样的使用体验：

- 支持单图、批量识别
- 支持 CSI / USB 摄像头实时识别
- 输出可视化结果和 `result.json`
- 默认使用本地模型目录，支持离线运行

## 目录结构

```text
work_badge/
├── img/                          # 原始测试图片
├── imgs/                         # 默认测试图片目录
├── models/                       # 本地零样本模型目录（可离线运行）
│   └── grounding-dino-tiny/
├── outputs/                      # 运行后输出结果
├── README.md
├── work_badge.py                 # 核心识别逻辑（含摄像头）
└── work_badge_test.py            # 命令行入口（含模型下载）
```

## 实现方案

由于 `yolov8n.pt` 不包含“工作牌”类别，当前采用两阶段“免训练”方案：

1. `yolov8n.pt` 先检测 `person`
2. 取每个人胸前区域作为 ROI
3. 使用本地 `Grounding DINO` 零样本模型在 ROI 内找 `employee badge / work badge / name tag / id card / lanyard badge`
4. 结合几何约束做误检过滤，输出 `with_work_badge` / `without_work_badge`

## 本地模型与离线运行

默认零样本模型路径：

- `work_badge/models/grounding-dino-tiny`

默认远程仓库（只用于下载阶段）：

- `IDEA-Research/grounding-dino-tiny`

运行时模型加载策略：

- 仅从本地目录加载（`local_files_only=True`）
- 本地模型不存在时直接报错
- 需要你先执行一次“下载到本地”动作

### 下载到本地（联网一次）

```bash
cd /Users/pursues/Desktop/project/huitian/python/recognize-algorithm-test/work_badge
python3 work_badge_test.py \
  --download-zero-shot-model \
  --prepare-only
```

下载完成后，后续可断网运行。

## 启动命令

单张图片检测：

```bash
cd /Users/pursues/Desktop/project/huitian/python/recognize-algorithm-test/work_badge
python3 work_badge_test.py \
  --image imgs/badge.png
```

批量检测：

```bash
cd /Users/pursues/Desktop/project/huitian/python/recognize-algorithm-test/work_badge
python3 work_badge_test.py \
  --image-dir imgs
```

实时识别（默认 CSI）：

```bash
cd /Users/pursues/Desktop/project/huitian/python/recognize-algorithm-test/work_badge
python3 work_badge_test.py --camera
```

实时识别（显式 CSI 参数）：

```bash
cd /Users/pursues/Desktop/project/huitian/python/recognize-algorithm-test/work_badge
python3 work_badge_test.py \
  --camera csi \
  --width 1280 \
  --height 720 \
  --framerate 30 \
  --flip-method 0
```

实时识别（USB）：

```bash
cd /Users/pursues/Desktop/project/huitian/python/recognize-algorithm-test/work_badge
python3 work_badge_test.py --camera usb
```

如果你把本地模型目录放在别处：

```bash
cd /Users/pursues/Desktop/project/huitian/python/recognize-algorithm-test/work_badge
python3 work_badge_test.py \
  --image imgs/badge.png \
  --zero-shot-model /absolute/path/to/grounding-dino-tiny
```

## 输出结果

运行后默认生成：

- `outputs/predict/`：带框可视化图片
- `outputs/result.json`：结构化结果

`result.json` 包含：

- 输入图片
- 本地零样本模型路径
- 人员框和胸前 ROI
- 工作牌框
- 每个人是否佩戴工作牌
- 总体识别结果

## 依赖

- `ultralytics`
- `torch`
- `transformers`
- `huggingface_hub`（仅下载本地模型时需要联网）
- `opencv-python`

### 边缘设备（离线/Jetson）部署说明

我们为你提供了 **零安装 (Zero-Install)** 的免折腾体验。你只需要将代码和 `wheels` 文件夹拷贝到 Jetson 边缘盒子中，即可直接运行，无需再敲任何 `pip install` 命令！

1. 将最新的 `work_badge` 目录（请确保包含 `wheels` 文件夹以及最新修改的 `*.py` 文件）拷贝到 Jetson。
2. 直接运行测试脚本（即使没有网络也可以）：
   ```bash
   python3 work_badge_test.py --image imgs/badge.png
   ```

**原理解释**：代码运行时，如果检测到本地尚未安装 `transformers` 库，会自动从 `wheels` 文件夹中解压出对应 ARM64 架构的离线安装包（`.whl` 是标准的 ZIP 文件）到临时的 `libs` 目录中，并直接在内存中加载，彻底绕过了容易报错的 `pip` 校验和安装流程。

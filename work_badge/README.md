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

由于不同操作系统的文件编码和 Python C 扩展（如 `tokenizers`）在跨平台直接拷贝解压时极易损坏（例如出现 `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb0`），**强烈建议不要直接使用拷贝的 `libs` 文件夹运行**。

我们已为你准备好了离线安装包（`.whl`），你只需在盒子上执行**一条离线安装命令**即可：

1. 将整个 `work_badge` 目录拷贝到 Jetson。
2. 在 Jetson 上进入 `work_badge` 目录并执行：
   ```bash
   pip3 install --no-index --find-links=wheels transformers huggingface_hub
   ```
3. 安装完成后，即可直接运行（即使没有网络也可以）：
   ```bash
   python3 work_badge_test.py --image imgs/badge.png
   ```

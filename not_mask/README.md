# not_mask

`not_mask` 用于识别“未戴口罩”的人脸，不需要自己训练模型。

当前实现采用“两阶段”方案：

- 第 1 步：使用 OpenCV 自带的人脸检测器找到人脸区域
- 第 2 步：把每张人脸裁剪后送入当前目录下的本地口罩分类模型，判断是 `WithoutMask` / `WithMask`，或兼容 `Face_Mask Not_Found` / `Face_Mask Found`

这样做的目标是尽量复用开源现成权重，快速落地一个可运行的未戴口罩识别模块。

## 目录结构

```text
not_mask/
├── imgs/                   # 测试图片
│   ├── mask.png
│   └── not-mask-leijun.jpg
├── models/
│   └── face-mask-detection/ # 本地预训练模型目录
├── outputs/                # 运行后自动生成，可按需删除
├── not_mask.py             # 未戴口罩识别核心模块
├── not_mask_test.py        # 命令行测试入口
└── README.md               # 当前说明文档
```

## 目标

- 输入单张图片或整个图片目录
- 自动定位人脸
- 对每张人脸输出是否“未戴口罩”
- 生成可视化结果图和 `result.json`
- 不做本地训练，直接复用已经放在当前目录下的本地预训练权重

## 默认模型

当前默认模型已经下载到当前目录下的本地路径：

- `models/face-mask-detection`

当前脚本默认直接加载本地 `models/face-mask-detection`，不依赖 `.hf-cache/`。
模型推理基于设备上已有的 `torch`、`torchvision`、`opencv-python`、`Pillow` 运行时。

为了减少环境复杂度，当前落地版本固定使用已经下载到本地目录的权重做人脸裁剪分类。

## 启动命令

建议直接进入 `not_mask/` 目录执行，这样整个目录单独拷走后也能直接测试：

```bash
cd not_mask
```

### 1. 批量检测 `not_mask/imgs` 下的全部图片

```bash
python3.11 not_mask_test.py
```

### 2. 仅检测默认示例图

```bash
python3.11 not_mask_test.py --default-image
```

### 3. 指定单张图片

```bash
python3.11 not_mask_test.py --image imgs/not-mask-leijun.jpg
```

### 4. 显式指定当前目录下的本地模型

```bash
python3.11 not_mask_test.py --model ./models/face-mask-detection
```

### 5. 调整阈值

```bash
python3.11 not_mask_test.py --score-threshold 0.7
```

### 6. 指定输出目录并关闭可视化图片保存

```bash
python3.11 not_mask_test.py --output-dir outputs --no-save-vis
```

### 7. USB 摄像头实时识别

```bash
python3.11 not_mask_test.py --camera usb
```

### 8. CSI 摄像头实时识别

```bash
python3.11 not_mask_test.py --camera csi
```

### 9. 指定数字摄像头序号

```bash
python3.11 not_mask_test.py --camera 0
```

常用实时识别参数：

```bash
python3.11 not_mask_test.py --camera csi --width 1280 --height 720 --framerate 30
python3.11 not_mask_test.py --camera usb --frame-log-interval 10
python3.11 not_mask_test.py --camera csi --warmup-frames 15 --max-failed-reads 100
```

## 轻量化说明

- 当前目录已经去掉 `.hf-cache/` 方案，模型改为本地 `models/face-mask-detection`
- 模型目录只保留推理必需文件
- `outputs/` 不是运行必需内容，可以随时删除，程序下次运行会自动重新生成
- 当前实现不再依赖 `transformers`、`regex`、`safetensors` 这类额外 Python 包

## 输出结果

- `not_mask/outputs/result.json`：保存每张图片的人脸检测与未戴口罩判定结果
- `not_mask/outputs/predict/*.jpg|*.png`：保存带框可视化结果
- 实时摄像头模式下会弹出窗口显示检测结果，按 `Q` 退出

`result.json` 中主要字段：

- `summary.total_images`：处理的图片数量
- `summary.total_faces`：检测到的人脸数量
- `summary.no_mask_faces`：判定为未戴口罩的人脸数量
- `items[].face_detections[]`：每张脸的分类标签、置信度、坐标和完整分数

## 注意事项

- 当前识别重点是“是否戴口罩”，不是严格意义上的“脸上完全无遮挡物”
- 如果图片里没有检测到人脸，脚本会退化为对整张图做一次分类，避免直接无结果
- 如果你后续想识别“墨镜、手挡脸、头发遮挡、口罩”等更广义的脸部遮挡，建议再换成专门的 face occlusion 模型
- `--camera csi` 使用的是和 `helmet` 一样的 GStreamer 管线，适合 Jetson 常见 CSI 摄像头
- 当前实现只依赖 `not_mask/` 目录自身内容，不再依赖外层项目目录；单独拷走该目录后，安装好依赖即可运行
- `--model` 现在只接受本地模型目录路径，不再支持在线仓库名
- 当前目录已经尽量自包含，但仍默认复用设备上已有的 `torch`、`torchvision`、`cv2`、`PIL`、`numpy` 运行时

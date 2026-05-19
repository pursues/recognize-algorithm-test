# work_clothes

## 目标

`work_clothes` 用于识别图片中的人员是否穿着反光背心/安全马甲。当前实现不需要自行训练，直接复用已经训练好的开源权重：

- `work_clothes/models/safety_vest_best.pt`

当前目标不再是泛化“工作服”，而是更明确的 PPE 场景：`反光背心 / 安全马甲 / Safety Vest`。

## 目录结构

```text
work_clothes/
├── imgs/                  # 测试图片
├── models/                # 已训练好的安全马甲权重
├── outputs/               # 运行后生成的可视化结果和 result.json
├── README.md              # 当前说明文档
├── work_clothes.py        # 核心识别逻辑
└── work_clothes_test.py   # 命令行测试入口
```

## 实现说明

当前流程：

1. 直接加载已经训练好的 `safety_vest_best.pt`
2. 输出模型检测到的 PPE 类别
3. 将 `person_with_vest`、`Safety Vest`、`Vest`、`Reflective-Jacket` 统一视为“穿了反光背心/安全马甲”
4. 将 `No Vest`、`NO-Safety Vest` 统一视为“未穿反光背心/安全马甲”

## 权重来源与可用性结论

这次我实际拉取并验证了几个带训练后权重的开源仓库：

- `ADiTyaRaj8969/Safety-Vest-and-Helmet-Detection`
  - 仓库内直接提供 `Q1/runs/detect/vest_helmet_final/weights/best.pt`
  - 类别定义明确：`No Vest`、`person_with_helmet`、`person_with_vest`
  - 我实际加载了该权重并读取了类别名，且对其测试集前 10 张图做了抽样验证，前 10 张图中和标签一致 7/10
  - 在你当前本地图片中：`clothe.png` 检出 `person_with_vest`，`no-phone-leijun.jpg` 检出 `No Vest`
  - 当前版本采用这个权重作为默认模型
- `ftnabil97/Construction-Site-Safety-Gears-Detection-Model-Yolov8`
  - 仓库内直接提供 `models/best.pt`
  - 类别定义包含 `Safety Vest` 与 `NO-Safety Vest`
  - 我实际加载了该权重并验证能正确读出类别名
  - 但在你当前 `clothe.png` 上更偏向输出 `NO-Safety Vest`，不如上面的权重贴合你现有样例
- `prodbykosta/ppe-safety-detection-ai`
  - 仓库内直接提供 `best.pt`
  - 我实际加载后发现权重真实类别是 8 类 PPE：`Boots/Ear-protection/Glass/Glove/Helmet/Mask/Person/Vest`
  - 但它和仓库中 `dataset/safety-Helmet-Reflective-Jacket/data.yaml` 的 2 类配置对不上，不建议直接作为这次默认方案

结论：当前默认选用的 `safety_vest_best.pt` 是“已经训练好的安全马甲权重”，并且我已经做过实际加载和样例验证，不是未实验的零样本方案。

## 启动命令

单张图片检测：

```bash
cd /Users/pursues/Desktop/project/huitian/python/recognize-algorithm-test/work_clothes
python3 work_clothes_test.py \
  --image imgs/clothe.png
```

批量检测 `imgs` 目录：

```bash
cd /Users/pursues/Desktop/project/huitian/python/recognize-algorithm-test/work_clothes
python3 work_clothes_test.py \
  --image-dir imgs
```

默认使用 CSI 摄像头实时识别：

```bash
cd /Users/pursues/Desktop/project/huitian/python/recognize-algorithm-test/work_clothes
python3 work_clothes_test.py --camera
```

显式指定 CSI 摄像头参数：

```bash
cd /Users/pursues/Desktop/project/huitian/python/recognize-algorithm-test/work_clothes
python3 work_clothes_test.py \
  --camera csi \
  --width 1280 \
  --height 720 \
  --framerate 30 \
  --flip-method 0
```

如果你想切到 USB 摄像头：

```bash
cd /Users/pursues/Desktop/project/huitian/python/recognize-algorithm-test/work_clothes
python3 work_clothes_test.py --camera usb
```

如果你想换成别的已训练权重：

```bash
cd /Users/pursues/Desktop/project/huitian/python/recognize-algorithm-test/work_clothes
python3 work_clothes_test.py \
  --image imgs/clothe.png \
  --ppe-model models/safety_vest_best.pt \
  --conf 0.25 \
  --iou 0.45
```

## 输出结果

运行后默认会生成：

- `outputs/predict/`：带框可视化图片
- `outputs/result.json`：结构化识别结果

其中 `result.json` 里会包含：

- 每个检测框的位置
- 原始类别名
- 是否判定为反光背心/安全马甲
- 当前检测状态，如 `with_safety_vest`、`without_safety_vest`

## 实时识别说明

- `--camera` 不带值时默认按 `csi` 处理
- CSI 摄像头通过 GStreamer `nvarguscamerasrc` 打开，适合 Jetson 设备
- 实时窗口按 `Q` 退出
- 终端会按 `--frame-log-interval` 周期输出检测日志

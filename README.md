# 物联网盒子识别算法测试

安全帽模型与算法测试

当前 `helmet` 功能已经改成独立目录运行，模型、测试图片、脚本和输出结果都放在 `helmet` 目录下。
把整个 `helmet` 目录单独拷走后，只要 Python 环境里安装了 `ultralytics`，也可以独立运行，不再依赖当前项目下的 `framework`、`feature_registry` 等公共代码。

目录结构示例：

```text
helmet/
├── best.pt
├── helmet.py
├── helmet-test.py
├── imgs/
│   └── helmet.jpg
└── outputs/
```

首次使用建议安装依赖：

```bash
pip install ultralytics
```

进入 `helmet` 目录后执行：

```bash
cd /Users/pursues/Desktop/project/huitian/python/recognize-algorithm-test/helmet
python3 helmet-test.py
```

默认行为：

- 默认读取当前目录下的 `best.pt`
- 默认读取当前目录下 `imgs/helmet.jpg`
- 默认在当前目录下创建 `outputs`
- 默认生成可视化结果图，保存到 `outputs/predict`
- 默认生成检测结果 JSON，保存到 `outputs/result.json`

可选参数示例：

```bash
python3 helmet-test.py --image imgs/helmet.jpg --model best.pt
python3 helmet-test.py --output-dir outputs
python3 helmet-test.py --no-save-vis
```





## 1. 说明

你给出的模型页面是：

- `https://huggingface.co/keremberke/yolov8n-hard-hat-detection`

根据页面说明，这个模型本身并不是“物联网盒子识别”模型，而是一个基于 YOLOv8 的目标检测模型，支持的类别只有：

```python
['Hardhat', 'NO-Hardhat']
```

页面给出的公开指标是：

- `mAP@0.5(box) = 0.836`
- 说明：这是作者在其验证集上的自报结果（self-reported）

因此，如果你的目标是“识别物联网盒子”，这个模型可以作为**测试流程参考**，但不能直接说明它对“物联网盒子”有效。  
如果要严肃评估“物联网盒子识别率”，应当使用：

- 面向“物联网盒子”类别的数据集
- 或重新训练/微调一个“盒子检测模型”

---

## 2. 模型页给出的基本用法

模型页建议安装：

```bash
pip install ultralyticsplus==0.0.24 ultralytics==8.0.23
```

基础推理示例：

```python
from ultralyticsplus import YOLO, render_result

# 加载 Hugging Face 模型
model = YOLO("keremberke/yolov8n-hard-hat-detection")

# 设置推理参数
model.overrides["conf"] = 0.25
model.overrides["iou"] = 0.45
model.overrides["agnostic_nms"] = False
model.overrides["max_det"] = 1000

# 单张图片预测
image = "test.jpg"
results = model.predict(image)

print(results[0].boxes)

# 可视化
render = render_result(model=model, image=image, result=results[0])
render.show()
```

这个示例只能说明“模型能跑起来”，还不能说明“识别率和有效性达标”。

---

## 3. 如何测试识别率

要测试一个识别算法，至少要分成两部分：

1. **离线准确率测试**：看识别是否准
2. **在线有效性测试**：看实际部署时是否好用

### 3.1 准备测试集

建议单独准备一套**从未参与训练**的测试集，不能拿训练图片直接评估。

如果你测试的是“物联网盒子识别”，建议测试集覆盖以下情况：

- 正常光照
- 弱光/背光
- 遮挡
- 不同距离
- 不同角度
- 复杂背景
- 多目标同时出现
- 小目标场景
- 模糊场景
- 误检干扰物（长得像盒子的设备、纸箱、网关、路由器等）

建议数量：

- 最低：`200~300` 张
- 较可靠：`800~2000` 张

如果是视频场景，建议额外抽帧形成测试图片，避免只在静态照片上评估。

### 3.2 人工标注 Ground Truth

测试集需要人工标注真实框（Ground Truth），格式可选：

- YOLO 格式
- COCO 格式

如果你的目标是“物联网盒子”，至少需要一个类别，例如：

```python
['iot_box']
```

如果后续还要区分型号，可以扩展为：

```python
['iot_box_a', 'iot_box_b', 'iot_box_c']
```

### 3.3 评估核心指标

建议重点关注以下指标：

- `Precision`：预测为目标时，有多少是真的
- `Recall`：真实目标里，有多少被识别出来
- `F1-score`：Precision 和 Recall 的综合指标
- `mAP@0.5`：IoU=0.5 时的检测平均精度
- `mAP@0.5:0.95`：更严格、更常用的综合指标
- `False Positive`：误检数量
- `False Negative`：漏检数量

简单理解：

- 误检多：系统会“乱报”
- 漏检多：系统会“看不见”
- 仅看 mAP 不够，还要看误检率和漏检率

### 3.4 建议的合格标准

可以按业务要求先设一个基线，例如：

- `Precision >= 0.90`
- `Recall >= 0.85`
- `mAP@0.5 >= 0.90`
- 关键场景漏检率 `< 5%`

如果是安防、巡检、告警类业务，通常更应关注：

- 漏检率是否可接受
- 误报是否会造成大量无效告警

---

## 4. 如何测试“有效性”

“有效性”不只是准确率，还包括是否适合实际场景。

建议从以下几个方面测试：

### 4.1 场景有效性

检查模型在真实业务场景是否稳定：

- 摄像头安装高度变化后是否还能识别
- 远距离小目标是否还能识别
- 设备部分遮挡时是否还能识别
- 夜间或低照度下是否明显下降
- 背景复杂时是否出现大量误检

### 4.2 时延有效性

如果要部署在边缘侧、盒子端或低算力设备，必须测试：

- 单张推理耗时
- 平均 FPS
- CPU 占用
- 内存占用
- 模型加载时间

建议记录三种环境：

- 本地 PC
- 目标服务器
- 实际部署设备（如边缘网关、工控机、嵌入式盒子）

### 4.3 稳定性有效性

建议做长时间运行测试，例如：

- 连续运行 `1~4` 小时
- 检查是否出现内存持续增长
- 检查是否出现推理速度下降
- 检查视频流断开重连后是否恢复正常

### 4.4 业务有效性

最终要回答的是：这个算法是否真的帮业务解决问题？

例如：

- 是否减少人工巡检工作量
- 是否降低漏报率
- 是否能在可接受时间内告警
- 是否支持后续追溯和截图留存

---

## 5. 推荐测试流程

推荐按下面步骤执行：

### 步骤 1：先做基线跑通

先按模型页给的示例确认：

- 环境可安装
- 模型可下载
- 单张图片可推理
- 结果可视化正常

### 步骤 2：准备业务测试集

针对你的真实目标准备测试集：

- 如果你要测安全帽识别，就用安全帽图片
- 如果你要测物联网盒子识别，就必须用盒子图片

### 步骤 3：批量推理

对测试集全部图片批量预测，保存：

- 预测框
- 置信度
- 可视化结果图

### 步骤 4：计算指标

拿预测结果和人工标注结果对比，计算：

- Precision
- Recall
- F1
- mAP@0.5
- mAP@0.5:0.95

### 步骤 5：分析错误案例

重点整理：

- 哪些图片误检
- 哪些图片漏检
- 是否集中在某些场景
- 是否是某些角度、尺寸、光照导致问题

### 步骤 6：做部署有效性测试

在真实设备上验证：

- 速度是否够
- 资源占用是否可接受
- 长时间运行是否稳定

---

## 6. 一个可执行的测试方案

如果你只是想快速验证这个模型页面里的算法测试方法，可以按下面做。

### 6.1 安装依赖

```bash
pip install ultralyticsplus==0.0.24 ultralytics==8.0.23
```

### 6.2 编写快速测试脚本

保存为 `quick_test.py`：

```python
from pathlib import Path
from ultralyticsplus import YOLO

model = YOLO("keremberke/yolov8n-hard-hat-detection")
model.overrides["conf"] = 0.25
model.overrides["iou"] = 0.45
model.overrides["agnostic_nms"] = False
model.overrides["max_det"] = 1000

image_dir = Path("test_images")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

for image_path in image_dir.glob("*"):
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
        continue

    results = model.predict(str(image_path), save=True, project=str(output_dir), name="predict")
    print(f"{image_path.name}: {results[0].boxes}")
```

运行：

```bash
python quick_test.py
```

这一步可以先观察：

- 是否能检测出目标
- 置信度是否合理
- 是否有明显误检/漏检

### 6.3 人工抽样复核

建议至少人工复核：

- `50` 张命中样本
- `50` 张未命中样本
- `50` 张复杂场景样本

重点看：

- 检测框是否框准
- 置信度高但其实错误的情况有多少
- 漏掉的目标是否集中在某一类场景

---

## 7. 如果目标是“物联网盒子识别”，应该怎么做

这是最关键的一点。

当前 Hugging Face 页面上的模型是“安全帽检测模型”，并不适合直接证明“物联网盒子识别效果”。  
如果你的项目标题是“物联网盒子识别算法测试”，建议改成下面流程：

### 方案 A：先验证测试流程

先用这个公开模型熟悉完整流程：

- 模型加载
- 图片推理
- 结果可视化
- 批量处理
- 指标计算思路

适合目的：

- 学会怎么测目标检测模型
- 打通技术链路

### 方案 B：换成物联网盒子数据和模型

真正要做“物联网盒子识别算法测试”，建议：

1. 自采集“物联网盒子”图片或视频
2. 标注 `iot_box` 类别
3. 使用 YOLOv8n / YOLOv8s 重新训练
4. 在独立测试集上评估准确率
5. 在实际设备上评估时延和稳定性

这是更合理、也更可信的方案。

---

## 8. 最终结论

基于该 Hugging Face 页面，可以得到以下结论：

- 该页面给出了模型推理方法和一个公开指标 `mAP@0.5 = 0.836`
- 这个结果只能说明模型在其原始任务（安全帽检测）上有一定效果
- 不能直接证明它对“物联网盒子识别”有效
- 如果你要测试“识别率”，必须准备独立测试集并计算 Precision、Recall、F1、mAP 等指标
- 如果你要测试“有效性”，还必须验证真实场景表现、误检漏检、速度、资源占用和稳定性

如果后续你愿意，我可以继续帮你补两部分内容：

1. 直接生成一个 `test.py`，用于批量跑图片并导出检测结果  
2. 再生成一个“评估脚本”，自动统计 Precision / Recall / mAP 所需数据

# 抽烟识别算法测试

当前项目的 `smoking` 功能已经改成独立目录运行，模型、测试图片、脚本和输出结果都放在 `smoking` 目录下。

进入 `smoking` 目录后，只要 Python 环境里安装了 `ultralytics`，就可以独立运行，不依赖当前项目下的其他公共代码。

## 目录结构

```text
smoking/
├── best.pt
├── smoking.py
├── smoking-test.py
├── imgs/
│   ├── smoking.jpeg
│   └── phone-game.jpg
└── outputs/
```

## 安装依赖

首次使用建议安装：

```bash
pip install ultralytics
```

如果需要实时摄像头识别，还建议安装：

```bash
pip install opencv-python
```

## 模型说明

当前接入的是开源抽烟检测权重 `Smoking_Detection.pt`。

该模型可直接识别三类目标：

- `cigarette`
- `face`
- `smoking`

其中结果里的 `detected` 只表示是否检测到：

- `cigarette`
- `smoking`

即使图片中只检测到 `face`，也不会被判定为抽烟。

## 图片识别

进入目录后执行：

```bash
cd /Users/pursues/Desktop/project/huitian/python/recognize-algorithm-test/smoking
python3 smoking-test.py
```

默认行为：

- 默认读取当前目录下的 `best.pt`
- 默认读取当前目录下的 `imgs/smoking.jpeg`
- 默认在当前目录下创建 `outputs`
- 默认生成可视化结果图，保存到 `outputs/predict`
- 默认生成检测结果 JSON，保存到 `outputs/result.json`

常用示例：

```bash
python3 smoking-test.py --image imgs/smoking.jpeg
python3 smoking-test.py --image imgs/phone-game.jpg
python3 smoking-test.py --model best.pt
python3 smoking-test.py --output-dir outputs
python3 smoking-test.py --no-save-vis
```

## 实时识别

USB 摄像头：

```bash
python3 smoking-test.py --camera usb
```

CSI 摄像头：

```bash
python3 smoking-test.py --camera csi
```

指定摄像头序号：

```bash
python3 smoking-test.py --camera 0
```

常用可选参数：

```bash
python3 smoking-test.py --camera usb --frame-log-interval 10
python3 smoking-test.py --camera csi --width 1280 --height 720 --framerate 30
python3 smoking-test.py --camera csi --warmup-frames 15 --max-failed-reads 100
```

运行后会：

- 打开摄像头实时识别
- 在窗口中显示检测框
- 每隔若干帧在终端打印一次检测结果
- 按 `Q` 退出

## 输出结果

图片识别完成后，结果会输出到：

- 可视化结果图：`smoking/outputs/predict`
- 检测结果 JSON：`smoking/outputs/result.json`

`result.json` 中主要字段说明：

- `detections`：模型输出的全部检测结果
- `smoking_detections`：只保留 `cigarette` 和 `smoking`
- `detected`：是否检测到抽烟相关目标
- `result_file`：结果文件路径

## 当前验证情况

当前已经使用目录中的测试图片完成验证：

- `imgs/smoking.jpeg`
  - 可检出 `face`
  - 可检出 `cigarette`
  - `detected = true`
- `imgs/phone-game.jpg`
  - 只检出 `face`
  - `detected = false`

这说明当前模型已经比单纯的烟雾检测模型更贴近“人手持香烟/抽烟动作”场景。

## 相关文件

- 核心实现：`smoking/smoking.py`
- 运行入口：`smoking/smoking-test.py`
- 测试图片目录：`smoking/imgs`
- 输出目录：`smoking/outputs`

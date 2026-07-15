# Drowsiness Detection Models

这个目录只保留了两个困倦检测模型的推理代码，适合直接上传到 GitHub。

- `model1`：MediaPipe + YOLO 分类 + 时序融合 + 状态机，功能更完整
- `model2`：MediaPipe + MobileViT，结构更轻

不包含的内容：

- 训练日志
- `__pycache__`
- 测试文件
- 本地模型权重

## 目录结构

```text
transmit/
├── README.md
├── requirements.txt
├── .gitignore
├── model1/
│   ├── app.py
│   ├── config.yaml
│   ├── models/
│   ├── scripts/
│   └── src/
└── model2/
    ├── app.py
    ├── models/
    └── scripts/
```

## 环境配置

推荐 Python `3.11`。

```bash
conda create -n drowsiness python=3.11 -y
conda activate drowsiness
pip install torch torchvision
pip install -r requirements.txt
```

如果你要用 GPU 版 PyTorch，把上面的 `pip install torch torchvision` 换成你自己 CUDA 对应的安装命令即可。

## 模型下载

### model1

必须手动准备：

```bash
wget -O model1/models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

`model1/models/best.pt` 可选：

- 如果本地已有权重，放到 `model1/models/best.pt`
- 如果没有，程序会在首次运行时自动从 Hugging Face 下载 `mosesb/drowsiness-detection-yolo-cls/best.pt`

### model2

`model2` 默认会自动下载并缓存：

- `mosesb/drowsiness-detection-mobileViT-v2`
- `models/face_landmarker.task`（仅在当前 MediaPipe 后端需要时下载）

如果你想离线运行，也可以提前把文件放到：

- `model2/models/mobilevit_drowsiness.pth`
- `model2/models/face_landmarker.task`

## 使用方式

这是本地摄像头实时推理项目，直接在有摄像头的机器上运行即可。

### model1

先检查模型初始化：

```bash
python model1/scripts/model_init_check.py
```

可选摄像头检查：

```bash
python model1/scripts/camera_smoke_test.py
```

启动：

```bash
python model1/app.py
```

### model2

先检查模型初始化：

```bash
python model2/scripts/model_init_check.py
```

启动：

```bash
python model2/app.py
```

## 上传 GitHub

直接上传整个 `transmit/` 即可。`.gitignore` 已经排除了本地权重、日志和缓存文件。

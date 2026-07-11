# Member 2: Human Sensing Module

本目录是 AttentiveSlides+ 项目中 **成员 2：Human Sensing** 的当前工作区。它负责通过笔记本摄像头估计用户的粗粒度视觉关注区域，并输出可观察的学习状态信号，供成员 3 的多模态融合和成员 4 的 UI/Tutor 使用。


## 当前状态

已完成：

- OpenCV 笔记本摄像头输入
- MediaPipe Face Landmarker 人脸关键点和 iris landmarks
- solvePnP head pose 估计，并处理 pitch 接近 180 度的等价翻转问题
- 3x3 coarse gaze grid 预测
- row-first 九宫格命名
- gaze grid 到 slide AOI 的映射
- 9 点 personalized calibration
- yawning / eye closure / head-down / screen-facing 检测
- learning-state weak signals 聚合
- 分模块诊断脚本
- gaze grid 准确率评估脚本
- gaze 数据采集脚本，用于迁移到 GPU 电脑训练更强模型

进行中：

- 已采集 gaze 方向训练数据，正在迁移到有 GPU 的电脑训练更高准确率模型
- GPU 模型权重尚未回传；当前线上推理仍使用 calibration profile + standardized weighted centroid baseline

当前本地数据概况：

- `data/calibration_profiles/demo_user.json`：已有 personalized calibration profile
- `data/gaze_datasets/`：已有若干组 gaze labeled data
- `data/models/face_landmarker.task`：MediaPipe Face Landmarker 模型文件

## Grid 命名

Human Sensing 输出统一使用 row-first 命名：

```text
top_left       top_center       top_right
middle_left    middle_center    middle_right
bottom_left    bottom_center    bottom_right
```

旧的 col-first 名字，例如 `left_top`、`right_middle`，在代码里仍会被兼容映射到 `top_left`、`middle_right`，但新的输出、calibration profile 和 evaluation report 都使用 row-first 命名。



## 模块职责

### `webcam_capture.py`

统一相机输入层。

- `opencv`：当前 Windows 笔记本摄像头使用
- `realsense`：保留 RealSense 接入口

核心接口：

```python
CameraConfig(...)
create_camera_source(config)
```

### `gaze_estimator.py`

负责人脸关键点、head pose、gaze features、gaze grid 和 AOI 映射。

核心接口：

```python
extract_face_landmarks(frame) -> FaceLandmarks
estimate_head_pose(face_landmarks) -> HeadPose
predict_gaze_grid(frame, calibration_profile, slide_id) -> GazePrediction
map_gaze_to_aoi(gaze_prediction, aois) -> AOIPrediction
```

说明：

- 未校准时使用 heuristic baseline
- 校准后使用 `standardized_weighted_centroid`
- 后续 GPU 模型回传后，可在这里接入 learned gaze classifier

### `calibration.py`

负责 9 点 calibration target、采样、profile 构建和存储。

当前 profile 中包含：

- grid centroids
- grid spreads
- feature normalization
- feature weights
- 每个 grid 的 sample count

### `face_state_detector.py`

检测可观察学习状态信号：

- face detected
- screen-facing score
- yawn detected
- yawn count in last 3 minutes
- eye closure
- head down
- mouth aspect ratio
- eye aspect ratio

说明：当前的 `yawning` 和 `eye closure` 不是额外训练的 CNN、FER 或疲劳检测模型，而是基于 MediaPipe Face Landmarker 输出的人脸关键点做几何规则判断。

- `yawning`：使用 mouth aspect ratio，当前阈值为 `yawn_mar_threshold = 0.33`，并要求持续至少 `yawn_min_duration_sec = 0.6`
- `eye closure`：使用 eye aspect ratio，当前阈值为 `eyes_closed_ear_threshold = 0.19`

这种 rule-based detector 适合当前 MVP：可解释、轻量、容易调参，也方便在 demo 中说明每个 weak signal 从哪里来。后续如果需要更高准确率，可以再用已采集数据训练或接入专门的 yawn / eye-closure classifier，把这部分替换或增强。

### `learning_state_aggregator.py`

将 face-state 和 gaze history 聚合成 weak learning-state signals：

- `fatigue_signal_score`
- `possible_review_needed`
- `repeated_attention_to_same_aoi`

注意：这里不声称知道用户真实情绪、真实注意力或真实认知状态，只输出 observable weak signals。

## 安装与环境

推荐 Python 3.10。

```powershell
cd D:\semester7_summer\Human_centeredAI\human_sensing
conda activate human_ai
pip install -r requirements-human-sensing.txt
```

当前 Windows 环境已使用：

- `opencv-python`
- `mediapipe`
- `numpy`
- `pyrealsense2`
- `PyYAML`

MediaPipe Tasks API 需要模型文件：

```text
data/models/face_landmarker.task
```

## 分模块测试

这些脚本会一直运行，按 `q` / `Esc` 或 `Ctrl+C` 结束。

### 摄像头

```powershell
python scripts\diagnose_camera.py --camera-source opencv --device-index 0 --preview
```

### Face Landmarks

```powershell
python scripts\diagnose_landmarks.py --camera-source opencv --device-index 0 --mediapipe-model data\models\face_landmarker.task --preview
```

### Head Pose

```powershell
python scripts\diagnose_head_pose.py --camera-source opencv --device-index 0 --mediapipe-model data\models\face_landmarker.task --preview
```

正对屏幕时应重点看：

- raw pitch 可能接近 `180`
- normalized / official pitch 应接近 `0`

### Gaze

```powershell
python scripts\diagnose_gaze.py --camera-source opencv --device-index 0 --mediapipe-model data\models\face_landmarker.task --calibration-profile data\calibration_profiles\demo_user.json --preview
```

### Face State

```powershell
python scripts\diagnose_face_state.py --camera-source opencv --device-index 0 --mediapipe-model data\models\face_landmarker.task --preview
```

## Calibration

重新采集 personalized 9 点校准：

```powershell
python scripts\run_calibration.py --user-id demo_user --camera-source opencv --device-index 0 --mediapipe-model data\models\face_landmarker.task --prepare-sec 5.0 --warmup-sec 1.5 --dwell-sec 2.0
```

检查 profile：

```powershell
python scripts\diagnose_calibration_profile.py --calibration-profile data\calibration_profiles\demo_user.json
```

## 评估 Gaze Grid 准确率

```powershell
python scripts\evaluate_gaze_grid.py --camera-source opencv --device-index 0 --mediapipe-model data\models\face_landmarker.task --calibration-profile data\calibration_profiles\demo_user.json --prepare-sec 5.0 --warmup-sec 1.5 --dwell-sec 2.0
```

输出保存到：

```text
data/evaluation/
```

包括：

- frame-level accuracy
- valid-frame accuracy
- majority grid accuracy
- horizontal / vertical accuracy
- confusion statistics

## 采集 Gaze 训练数据

用于后续迁移到 GPU 电脑训练更高准确率的模型。

快速采一组约 2 分钟数据：

```powershell
python scripts\collect_gaze_dataset.py --user-id demo_user --repetitions 4 --save-frames
```

更短试采：

```powershell
python scripts\collect_gaze_dataset.py --user-id demo_user --repetitions 2 --save-frames
```

说明：

- 默认使用 OpenCV 摄像头 `0`
- 默认自动找 `data\models\face_landmarker.task`
- 默认 `prepare-sec=5.0`
- 默认 `warmup-sec=1.5`
- 默认 `dwell-sec=2.0`
- warmup 阶段目标窗口会嵌入摄像头预览
- recording 阶段只显示红点，避免干扰注视

输出目录：

```text
data/gaze_datasets/demo_user_YYYYMMDD_HHMMSS/
├── metadata.json
├── samples.jsonl
└── frames/
```

`samples.jsonl` 每行包含：

- `target_grid`
- `target_screen_xy`
- `head_pose`
- `features`
- `frame_path`
- `face_detected`
- `iris_available`

当前 GPU 训练计划：

- 数据从 `data/gaze_datasets/` 迁移到 GPU 电脑
- 训练 9 类 gaze grid classifier
- 类别为 `top_left` 到 `bottom_right`
- 训练完成后回传模型权重、label map 和 config
- 回传后再接入 `GazeEstimator`

建议训练输出：

```text
models/gaze_grid_model.pt
models/gaze_grid_label_map.json
models/gaze_grid_config.json
training_report.json
confusion_matrix.png
```

## 与 Member 1 的对接

Member 1 代码位于：

```text
AIAA3800HCAI/member1/
```

Member 1 已实现 Slide & AOI 模块，关键文件：

```text
AIAA3800HCAI/member1/modules/slide/slide_parser.py
AIAA3800HCAI/member1/modules/slide/aoi_manager.py
AIAA3800HCAI/member1/modules/slide/ocr.py
```

Member 1 的 AOI 输出接口：

```python
from modules.slide.aoi_manager import AOIManager

aoi_manager = AOIManager()
payload = aoi_manager.get_gaze_payload(deck_id, slide_id)
```

输出格式：

```json
{
  "slide_id": 2,
  "aois": [
    {
      "aoi_id": "left_block",
      "bbox": [0.05, 0.35, 0.48, 0.85],
      "type": "text"
    }
  ]
}
```

这个格式与 Member 2 的 `map_gaze_to_aoi(gaze_prediction, aois)` 在 schema 层兼容；当前已通过 `tests/test_human_sensing_logic.py` 中的 contract test 验证。注意：这表示数据结构可以直接对接，不表示 member1 的 live slide state 已经接入 member2 demo。

### Member 2 使用 Member 1 AOI 的方式

Member 2 需要从 Member 1 获取：

- `slide_id`
- `aois`
- 每个 AOI 的 `aoi_id`
- 每个 AOI 的归一化 `bbox = [x1, y1, x2, y2]`
- 每个 AOI 的 `type`

然后：

```python
gaze_prediction = estimator.predict(
    frame,
    calibration_profile=profile,
    slide_id=payload["slide_id"],
    face_landmarks=landmarks,
    head_pose=head_pose,
)

aoi_prediction = map_gaze_to_aoi(
    gaze_prediction,
    payload["aois"],
)
```

Member 2 输出给 Member 3：

```json
{
  "timestamp": 1710000000.32,
  "gaze_prediction": {
    "slide_id": 2,
    "gaze_grid": "middle_right",
    "predicted_aoi_id": "right_visual_region",
    "confidence": 0.72,
    "stable_duration_sec": 2.3
  },
  "learning_state": {
    "face_detected": true,
    "screen_facing_score": 0.86,
    "yawn_detected": false,
    "yawn_count_last_3min": 0,
    "eyes_closed": false,
    "eye_closure_duration_sec": 0.0,
    "head_down": false,
    "fatigue_signal_score": 0.23,
    "possible_review_needed": false
  }
}
```

## Demo Pipeline

当前可用示例 AOI：

```text
examples/sample_aois.json
```

运行完整 member2 demo：

```powershell
python scripts\run_human_sensing_demo.py --camera-source opencv --device-index 0 --mediapipe-model data\models\face_landmarker.task --calibration-profile data\calibration_profiles\demo_user.json --slide-id 5 --aoi-json examples\sample_aois.json
```

未来做运行时集成时，`--aoi-json` 应替换为 Member 1 当前 slide 的 `get_gaze_payload(deck_id, slide_id)` 输出，或由 UI/backend 直接把该 payload 传入 member2 的 `map_gaze_to_aoi()`。

## 对外接口汇总

```python
extract_face_landmarks(frame) -> FaceLandmarks
estimate_head_pose(face_landmarks) -> HeadPose
predict_gaze_grid(frame, calibration_profile, slide_id) -> GazePrediction
map_gaze_to_aoi(gaze_prediction, aois) -> AOIPrediction
detect_learning_state(frame, face_landmarks, history) -> LearningState
```


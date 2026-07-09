# Human-Centered AI Slide Learning Assistant

本项目是一个 Human-Centered AI 学习辅助系统。它不是四个孤立功能的拼接，而是围绕一个完整学习闭环设计：

```text
课件内容结构化
-> 摄像头理解用户视觉注意力和学习状态
-> 语音理解用户学习意图
-> 多模态融合确定用户想问的 slide 区域
-> AI Tutor 生成可解释、可纠正、可适应的学习反馈
```

当前仓库采用按成员分文件夹的结构，其中 `member1/` 文件夹负责 **Slide & AOI 模块**。

## 四人分工

| 成员 | 模块 | 核心任务 |
| --- | --- | --- |
| 成员 1 | Slide & AOI | 课件解析、slide 渲染、OCR 文本提取、AOI 区域划分、AOI 手动修正 |
| 成员 2 | Human Sensing | gaze/head pose、face detection、yawning、eye closure、learning-state signals |
| 成员 3 | Voice & Multimodal Fusion | 语音转文字、intent 识别、指代消解、多模态融合、adaptive strategy |
| 成员 4 | Tutor Agent & Integration | LLM Tutor、上下文检索、UI 集成、日志记录、实验评估 |

## 完整系统流程

### Step 1: 成员 1 输出 Slide 和 AOI

成员 1 把 PDF slide 转换成系统可以理解的数据：

```json
{
  "slide_id": 5,
  "slide_image_path": "data/slide_images/slide_005.png",
  "ocr_text": "This slide explains SHAP values...",
  "aois": [
    {
      "aoi_id": "right_figure",
      "bbox": [0.55, 0.18, 0.95, 0.78],
      "type": "figure",
      "text": "SHAP force plot"
    }
  ]
}
```

传给：

- 成员 2：用于 gaze 到 AOI 的映射。
- 成员 4：用于 Tutor 上下文检索。

### Step 2: 成员 2 输出 Gaze 和 Learning-State Signals

成员 2 通过摄像头估计用户正在看哪里，并检测可观察学习状态信号。

注意：系统不声称准确识别真实情绪或真实认知状态，只检测 observable learning-state signals。

```json
{
  "gaze_prediction": {
    "slide_id": 5,
    "gaze_grid": "right_middle",
    "predicted_aoi_id": "right_figure",
    "confidence": 0.72,
    "stable_duration_sec": 2.3
  },
  "learning_state": {
    "face_detected": true,
    "screen_facing_score": 0.86,
    "yawn_detected": false,
    "eyes_closed": false,
    "fatigue_signal_score": 0.23,
    "possible_review_needed": false
  }
}
```

传给：

- 成员 3：用于多模态融合。
- 成员 4：用于 UI 显示。

### Step 3: 成员 3 输出 Resolved Query

成员 3 理解用户说了什么，并结合 gaze、AOI 和 learning-state 判断用户真正想问哪个区域。

例如用户看着右边图说“解释这个”，系统解析为：

```json
{
  "query_id": "q_001",
  "slide_id": 5,
  "transcript": "解释这个",
  "intent": "explain",
  "resolved_aoi_id": "right_figure",
  "target_confidence": 0.74,
  "needs_confirmation": true,
  "adaptive_strategy": "normal",
  "evidence": [
    "用户使用了指代词：这个",
    "gaze_grid = right_middle",
    "predicted_aoi = right_figure"
  ],
  "alternative_targets": [
    {
      "aoi_id": "right_figure",
      "score": 0.74
    },
    {
      "aoi_id": "bottom_caption",
      "score": 0.51
    }
  ]
}
```

传给：

- 成员 4：用于生成 Tutor 回答。

### Step 4: 成员 4 输出 Tutor Response

成员 4 基于 slide context、resolved AOI、intent 和 adaptive strategy 生成回答。

```json
{
  "query_id": "q_001",
  "answer": "这个图展示的是 SHAP 如何解释模型预测结果...",
  "active_recall_question": "如果一个特征的 SHAP value 为正，它通常表示什么？",
  "adaptive_suggestion": null
}
```

显示给用户。

## 各成员接口设计

### 成员 1: Slide & AOI

负责文件：

```text
modules/slide/slide_parser.py
modules/slide/aoi_manager.py
modules/slide/ocr.py
data/aoi_manifest.json
data/slide_images/
```

对外接口：

```python
load_deck(pdf_path) -> deck_id
render_slide(deck_id, slide_id) -> slide_image_path
get_slide_aois(deck_id, slide_id) -> list[dict]
get_slide_text(deck_id, slide_id) -> str
update_aoi(deck_id, slide_id, aoi_id, bbox, aoi_type, text) -> dict
```

### 成员 2: Human Sensing

计划负责文件：

```text
modules/human_sensing/webcam_capture.py
modules/human_sensing/gaze_estimator.py
modules/human_sensing/calibration.py
modules/human_sensing/face_state_detector.py
modules/human_sensing/learning_state_aggregator.py
```

计划接口：

```python
extract_face_landmarks(frame) -> FaceLandmarks
estimate_head_pose(face_landmarks) -> HeadPose
predict_gaze_grid(frame, calibration_profile) -> GazePrediction
map_gaze_to_aoi(gaze_prediction, aois) -> AOIPrediction
detect_learning_state(frame, face_landmarks, history) -> LearningState
```

### 成员 3: Voice & Multimodal Fusion

计划负责文件：

```text
modules/interaction/speech_to_text.py
modules/interaction/intent_parser.py
modules/interaction/reference_resolver.py
modules/interaction/adaptive_policy.py
modules/interaction/interaction_history.py
```

计划接口：

```python
transcribe_audio(audio_path) -> Transcript
parse_intent(transcript) -> IntentResult
detect_deictic_reference(transcript) -> bool
resolve_reference(intent_result, gaze_prediction, learning_state, aois, history) -> ResolvedQuery
select_adaptive_strategy(learning_state, intent_result, history) -> AdaptiveStrategy
```

MVP intent 类型：

```text
explain
compare
quiz
summarize
simplify
step_by_step
review
break
```

### 成员 4: Tutor Agent & System Integration

计划负责文件：

```text
modules/tutor/context_retriever.py
modules/tutor/llm_tutor.py
modules/tutor/prompt_template.py
modules/logging/interaction_logger.py
app.py
evaluation/eval_aoi_accuracy.py
evaluation/eval_learning_state.py
evaluation/eval_usability.py
```

计划接口：

```python
retrieve_context(deck_id, slide_id, resolved_aoi_id, history) -> TutorContext
generate_tutor_response(tutor_context, intent, adaptive_strategy) -> TutorResponse
log_interaction(event) -> None
render_ui_state(slide, aois, gaze, learning_state, resolved_query, response) -> None
```

## member1 文件夹: Slide & AOI 模块

`member1/` 文件夹已经实现成员 1 的 Slide & AOI 系统。

### 当前保留代码

```text
member1/main_modular.py
member1/modules/slide/__init__.py
member1/modules/slide/slide_parser.py
member1/modules/slide/aoi_manager.py
member1/modules/slide/ocr.py
member1/requirements.txt
member1/data/
```

各文件作用：

- `member1/main_modular.py`：本地命令行测试入口。
- `member1/modules/slide/slide_parser.py`：加载 PDF、保存 deck 元数据、渲染 slide 图片、提取 PDF 原生文本。
- `member1/modules/slide/aoi_manager.py`：生成 AOI、合并文本块、保存 AOI manifest、手动新增/修改/删除 AOI、输出 gaze/tutor payload。
- `member1/modules/slide/ocr.py`：EasyOCR fallback，以及统一的 `TextBox` 数据结构。
- `member1/data/`：运行时数据目录，保存上传 PDF、渲染图片、deck metadata 和 AOI manifest。

## 安装依赖

```bash
cd /Users/herry/code/slide_aoi_system_fixed/member1
pip install -r requirements.txt
```

## 本地运行

处理 PDF 的第 2 页：

```bash
cd /Users/herry/code/slide_aoi_system_fixed/member1
python main_modular.py /Users/herry/slide_aoi_system/test_slides.pdf --slide-id 2
```

运行后会输出 JSON，并保存：

```text
data/uploaded_decks/{deck_id}.pdf
data/slide_images/{deck_id}_slide_002.png
data/deck_metadata.json
data/aoi_manifest.json
```

## 成员 1 核心代码用法

### 加载 PDF

```python
from modules.slide.slide_parser import SlideParser

parser = SlideParser()
deck_id = parser.load_deck("/path/to/slides.pdf")
```

### 渲染 slide

```python
image_path = parser.render_slide(deck_id, slide_id=2)
```

### 处理 slide 并生成 AOI

```python
from modules.slide.aoi_manager import AOIManager

aoi_manager = AOIManager()
slide_data = aoi_manager.process_slide(deck_id, slide_id=2)
```

输出结构：

```json
{
  "slide_id": 2,
  "slide_image_path": "data/slide_images/{deck_id}_slide_002.png",
  "ocr_text": "Elements in NLP...",
  "slide_text": "Elements in NLP...",
  "text_source": "pdf_text",
  "auto_aoi_method": "pdf_text_semantic",
  "aois": []
}
```

### 获取 slide 文本

```python
text = aoi_manager.get_slide_text(deck_id, slide_id=2)
```

### 获取 AOI 列表

```python
aois = aoi_manager.get_slide_aois(deck_id, slide_id=2)
```

### 手动修改 AOI

```python
updated = aoi_manager.update_aoi(
    deck_id=deck_id,
    slide_id=2,
    aoi_id="right_block",
    bbox=[0.52, 0.35, 0.95, 0.85],
    aoi_type="mixed",
    text="Updated text"
)
```

### 手动新增 AOI

```python
new_aoi = aoi_manager.add_aoi(
    deck_id=deck_id,
    slide_id=2,
    aoi_id="center_table",
    bbox=[0.30, 0.30, 0.75, 0.70],
    aoi_type="table",
    text="Manually labeled table"
)
```

### 删除 AOI

```python
aoi_manager.delete_aoi(deck_id, slide_id=2, aoi_id="center_table")
```

## 成员 1 处理逻辑

```text
1. load_deck()
   加载 PDF，复制到 data/uploaded_decks/，生成 deck_id。

2. render_slide()
   使用 PyMuPDF 把指定页渲染成 PNG 图片。

3. extract_pdf_text_boxes()
   优先使用 PyMuPDF 读取 PDF 原生文本和 bbox。

4. EasyOCR fallback
   如果 PDF 原生文本太少，才使用 EasyOCR 从图片里识别文字。

5. generate_rule_aois()
   生成基础规则 AOI。

6. build_pdf_semantic_aois() 或 build_text_block_aois()
   PDF 文本生成 semantic AOI；
   OCR 文本生成普通 text block AOI。

7. populate_rule_aoi_text()
   把自动识别出的文本回填到规则 AOI。

8. save_slide_data()
   保存统一结果到 data/aoi_manifest.json。
```

## PyMuPDF 原生文本优先

系统优先使用 PyMuPDF 原生文本，而不是一开始就 OCR。

原因是很多 PPT 导出的 PDF 本身包含文字层。PyMuPDF 可以直接读取这些文字和位置，通常比 OCR 更准确、更快。

EasyOCR 只作为 fallback：

- PDF 是扫描件；
- PDF 页面是一整张图片；
- PyMuPDF 读不到足够文字。

## AOI 类型

规则 AOI：

- `title`：标题区域。
- `top_region`：上方正文区域。
- `left_block`：左侧内容区域。
- `right_block`：右侧内容区域，可能是文字、图或混合内容。
- `right_visual_region`：右侧视觉区域，用于粗略覆盖图表/示意图。
- `bottom_region`：底部区域，通常是页脚或说明。
- `whole_slide`：整页 slide。

自动 AOI：

- `pdf_semantic_block_*`：由 PDF 原生文本生成的语义 AOI。
- `ocr_text_block_*`：由 EasyOCR fallback 生成的 OCR 文本块 AOI。

## 输出给 gaze 模块

```python
payload = aoi_manager.get_gaze_payload(deck_id, slide_id=2)
```

输出：

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

gaze 模块只需要 `aoi_id`、`bbox` 和 `type`。

## 输出给 tutor 模块

```python
payload = aoi_manager.get_tutor_payload(deck_id, slide_id=2)
```

输出：

```json
{
  "slide_id": 2,
  "ocr_text": "Elements in NLP...",
  "aois": [
    {
      "aoi_id": "pdf_semantic_block_3",
      "type": "text",
      "text": "Semantics\nLiteral meanings"
    }
  ]
}
```

tutor 模块主要使用整页文本 `ocr_text`、每个 AOI 的 `type` 和 `text`。

## 集成时间安排

| 阶段 | 时间 | 目标 | 负责人 |
| --- | --- | --- | --- |
| Phase 1 | Day 1-2 | 确定题目、system claim、related work | 全体 |
| Phase 2 | Day 3-4 | 完成 slide viewer、AOI schema、OCR 初版 | 成员 1 |
| Phase 3 | Day 4-6 | 完成 gaze/head pose、yawn、eye closure 初版 | 成员 2 |
| Phase 4 | Day 6-7 | 完成 STT、intent parser、指代词检测 | 成员 3 |
| Phase 5 | Day 7-9 | 完成 reference resolver 和 adaptive policy | 成员 3 |
| Phase 6 | Day 9-11 | 完成 LLM Tutor 和 UI 集成 | 成员 4 |
| Phase 7 | Day 11-13 | 完成 pilot test，修 bug | 全体 |
| Phase 8 | Day 14+ | 正式实验、report、presentation、demo video | 全体 |

## 最终闭环

```text
成员 1：把 slide 变成结构化 AOI
  ↓
成员 2：检测用户正在看哪个区域，以及是否出现 yawning / eye closure 等学习状态信号
  ↓
成员 3：理解用户语音意图，并融合 gaze + AOI + learning-state 判断用户想问什么
  ↓
成员 4：调用 LLM Tutor 生成基于 slide 的解释，并在 UI 中展示、记录和评估
```

最终系统核心能力：

```text
用户看着 slide 的某个区域，说“解释这个”
-> 系统根据 gaze 判断目标 AOI
-> 根据 yawning / eye closure / screen-facing 等信号判断是否需要调整回答方式
-> 显示预测目标和置信度
-> 用户确认或修正
-> AI Tutor 生成 grounded explanation / quiz / summary / review suggestion
```

这使项目不仅是普通的 slide QA 工具，而是一个具有 Human-Centered AI 特征的学习辅助系统。

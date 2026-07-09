# Slide & AOI System

这是 Human-Centered AI 学习辅助系统中的 Slide & AOI 模块。它负责把用户上传的 PDF 课件转换成结构化数据，包括 slide 图片、文本内容、AOI 区域，以及给 gaze 模块和 tutor 模块使用的接口输出。

## 保留的有用代码

当前只保留 modular 版本：

```text
main_modular.py
modules/slide/__init__.py
modules/slide/slide_parser.py
modules/slide/aoi_manager.py
modules/slide/ocr.py
README.md
data/
```

各文件作用：

- `main_modular.py`：本地命令行测试入口。
- `modules/slide/slide_parser.py`：负责加载 PDF、保存 deck 元数据、渲染 slide 图片、提取 PDF 原生文本。
- `modules/slide/aoi_manager.py`：负责生成 AOI、合并文本块、保存 AOI manifest、手动新增/修改/删除 AOI、输出 gaze/tutor payload。
- `modules/slide/ocr.py`：负责 EasyOCR fallback，以及统一的 `TextBox` 数据结构。
- `data/`：运行时数据目录，保存上传 PDF、渲染图片、deck metadata 和 AOI manifest。

## 安装依赖

进入项目目录：

```bash
cd /Users/herry/code/slide_aoi_system_fixed
```

安装依赖：

```bash
pip install PyMuPDF easyocr pillow
```

## 本地运行

处理 PDF 的第 2 页：

```bash
cd /Users/herry/code/slide_aoi_system_fixed
python main_modular.py /Users/herry/slide_aoi_system/test_slides.pdf --slide-id 2
```

运行后会输出 JSON，并保存：

```text
data/uploaded_decks/{deck_id}.pdf
data/slide_images/{deck_id}_slide_002.png
data/deck_metadata.json
data/aoi_manifest.json
```

## 核心接口

### 加载 PDF

```python
from modules.slide.slide_parser import SlideParser

parser = SlideParser()
deck_id = parser.load_deck("/path/to/slides.pdf")
```

返回：

```python
deck_id: str
```

### 渲染 slide

```python
image_path = parser.render_slide(deck_id, slide_id=2)
```

返回：

```python
image_path: str
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

返回：

```python
str
```

### 获取 AOI 列表

```python
aois = aoi_manager.get_slide_aois(deck_id, slide_id=2)
```

返回：

```python
list[dict]
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

## 主要处理逻辑

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

## group、children 和 confidence

`pdf_semantic_block_*` 是合并后的学习单元，也就是 group。

`children` 是 group 内部由哪些原始文本框合并而来，主要用于 debug。

`group_confidence` 表示系统认为这些文本应该合并成一个 AOI 的置信度。

正式给其他模块时，一般不需要使用 `children`。

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

gaze 模块只需要：

- `aoi_id`
- `bbox`
- `type`

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

tutor 模块主要使用：

- 整页文本 `ocr_text`
- 每个 AOI 的 `type`
- 每个 AOI 的 `text`

## 当前满足的要求

当前版本已经满足成员 1 的核心要求：

- 支持上传 PDF slide；
- 使用 PyMuPDF 渲染 slide 图片；
- 支持 OCR 文本提取；
- 优先使用 PyMuPDF 原生文本，EasyOCR fallback；
- 生成规则 AOI；
- 生成 semantic/text block AOI；
- 支持手动修改 AOI；
- 支持新增 AOI；
- 支持删除 AOI；
- 保存统一 JSON 到 `data/aoi_manifest.json`；
- 给 gaze 模块提供 AOI 接口；
- 给 tutor 模块提供文本接口。

## 当前局限

`right_visual_region` 只是粗略视觉区域，不是真正的图像检测。

系统目前还不能精确自动识别：

- 单独的表格；
- 单独的公式；
- 单独的流程图；
- 图中每个组件的 bbox。

如果后续继续升级，下一步可以加 figure/table/formula detection。

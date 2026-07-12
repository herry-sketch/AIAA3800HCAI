# LLM-Guided AOI Upgrade

这个升级是在原来的 member1 Slide & AOI 模块上增加一个可选的 LLM/VLM AOI 生成层。

## 设计逻辑

原流程：

```text
PyMuPDF 原生文本优先
-> EasyOCR fallback
-> rule AOI
-> PDF/OCR text block AOI
```

升级后：

```text
PyMuPDF 原生文本优先
-> EasyOCR fallback
-> rule AOI
-> PDF/OCR text block AOI
-> 可选 LLM/VLM guided AOI
-> 如果 LLM 失败，自动 fallback 到原来的 rule + PDF/OCR AOI
```

所以这个升级不会破坏你已经能跑的 MVP。

## 显存要求

如果用云端 API：

```text
本机几乎不需要显存
```

如果自己在云服务器跑视觉语言模型：

```text
7B 量化模型：建议 12-16GB VRAM
7B 非量化/更稳：建议 24GB VRAM
13B 级别：建议 40GB+ VRAM
更大模型：80GB 更稳
```

课程项目建议：

```text
先用 7B 视觉模型或云端 API
不要一开始上很大的模型
```

## 新增文件

```text
member1/modules/slide/llm_aoi.py
```

## 修改文件

```text
member1/modules/slide/aoi_manager.py
member1/main_modular.py
```

## 环境变量

这个版本使用 OpenAI-compatible chat completions 接口。

例如你的云端模型 endpoint 是：

```text
http://your-server:8000/v1/chat/completions
```

运行前设置：

```bash
export SLIDE_AOI_LLM_ENDPOINT="http://your-server:8000/v1/chat/completions"
export SLIDE_AOI_LLM_MODEL="qwen2.5-vl-7b-instruct"
export SLIDE_AOI_LLM_API_KEY="your_key_if_needed"
```

如果 endpoint 不需要 key，可以不设置 `SLIDE_AOI_LLM_API_KEY`。

## 运行

不用 LLM，保持原版本：

```bash
cd /Users/herry/code/slide_aoi_system_fixed/member1
python main_modular.py /Users/herry/slide_aoi_system/test_slides.pdf --slide-id 2
```

启用 LLM AOI：

```bash
cd /Users/herry/code/slide_aoi_system_fixed/member1
python main_modular.py /Users/herry/slide_aoi_system/test_slides.pdf --slide-id 2 --use-llm-aoi
```

## 输出变化

成功使用 LLM 时：

```json
{
  "auto_aoi_method": "llm_guided_with_pdf_text_semantic_fallback",
  "llm_aoi_status": "used",
  "aois": [
    {
      "aoi_id": "llm_aoi_1",
      "bbox": [0.05, 0.05, 0.95, 0.18],
      "type": "title",
      "text": "Elements in NLP",
      "source": "llm_guided",
      "group_confidence": 0.91
    }
  ]
}
```

LLM 失败但 fallback 成功时：

```json
{
  "auto_aoi_method": "pdf_text_semantic",
  "llm_aoi_status": "fallback_used",
  "llm_aoi_error": "SLIDE_AOI_LLM_ENDPOINT is not configured"
}
```

这说明系统没有崩，而是退回到了原来的规则/PDF/OCR AOI。

## 为什么这样设计

LLM/VLM 的优势：

- 可以更好识别 figure/table/formula；
- 可以把视觉区域和文本解释合成更自然的学习单元；
- 可以减少固定 6 个 rule AOI 的局限。

保留 rule fallback 的原因：

- LLM 输出可能格式错；
- 云端模型可能超时；
- 显存不够时模型可能跑不起来；
- gaze 模块需要稳定 bbox，不能完全依赖不稳定输出。

所以最终策略是：

```text
LLM 提升质量
Rule/PDF/OCR 保证系统永远有可用输出
```

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

## 使用 Qwen / DashScope API

如果你想买 Qwen API，可以直接用阿里云 DashScope 的 OpenAI-compatible 接口。你不需要等组员云端 endpoint。

先在终端设置：

```bash
export DASHSCOPE_API_KEY="你的 DashScope API Key"
export SLIDE_AOI_LLM_MODEL="qwen-vl-plus"
```

然后运行：

```bash
cd /Users/herry/code/slide_aoi_system_fixed/member1
python main_modular.py /Users/herry/slide_aoi_system/test_slides.pdf --slide-id 2 --use-llm-aoi
```

代码会自动使用：

```text
https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
```

你也可以把模型换成你 DashScope 账号里可用的其他视觉模型，例如更强或更新的 Qwen-VL 模型。

注意：

- 不要把真实 API key 写进代码或提交到 GitHub。
- 默认模型是 `qwen-vl-plus`，适合先跑通。
- 默认会把 slide 图片最长边压到 `1280`，减少图片 token 成本。

## 使用付费 OpenAI API

如果你现在买 OpenAI API，不需要等组员云端 endpoint。直接设置 `OPENAI_API_KEY` 即可。

```bash
export OPENAI_API_KEY="sk-your-openai-api-key"
export SLIDE_AOI_LLM_MODEL="gpt-4o-mini"
```

然后运行：

```bash
cd /Users/herry/code/slide_aoi_system_fixed/member1
python main_modular.py /Users/herry/slide_aoi_system/test_slides.pdf --slide-id 2 --use-llm-aoi
```

注意：

- 不要把真实 API key 写进代码或提交到 GitHub。
- 如果模型名字不可用，可以把 `SLIDE_AOI_LLM_MODEL` 换成你 API 项目里可用的 vision-capable model。
- 默认会把 slide 图片最长边压到 `1280`，减少 token/图片成本。

## 使用自定义云端 endpoint

这个版本也支持 OpenAI-compatible chat completions 接口。

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

如果是组员自己部署的 endpoint，并且不需要 key，可以设置一个占位 key：

```bash
export SLIDE_AOI_LLM_API_KEY="local"
```

## 环境变量优先级

```text
API key:
SLIDE_AOI_LLM_API_KEY > DASHSCOPE_API_KEY / QWEN_API_KEY > OPENAI_API_KEY

Model:
SLIDE_AOI_LLM_MODEL > QWEN_MODEL > OPENAI_MODEL > qwen-vl-plus 或 gpt-4o-mini

Endpoint:
SLIDE_AOI_LLM_ENDPOINT > OPENAI_BASE_URL + /chat/completions > DashScope endpoint 或 OpenAI official endpoint
```

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
  "llm_aoi_error": "LLM AOI API is not configured. Set DASHSCOPE_API_KEY, QWEN_API_KEY, OPENAI_API_KEY, or SLIDE_AOI_LLM_API_KEY."
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

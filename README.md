# Podcast Transcriber

一个用于转录并总结播客音频内容的工具。支持使用OpenAI Whisper或whisper.cpp进行语音识别，pyannote进行说话人区分，以及多种模型生成文本摘要（Transformers或本地Qwen模型）。

## 功能

- 将播客音频转换为文本
- 支持两种转录引擎：OpenAI Whisper和更快的whisper.cpp
- 区分不同说话人
- 为转录内容生成摘要
- 支持两种摘要生成模型：
  - Hugging Face Transformers (如BART, T5等)
  - 本地Qwen3-1.7B模型（更好的中文摘要效果）
- 新增: 使用MLX框架的Qwen模型（专为Apple Silicon优化）

## 安装

基本安装:
```bash
pip install -e .
```

如果需要使用whisper.cpp转录引擎:
```bash
# NVIDIA GPU support
GGML_CUDA=1 pip install git+https://github.com/absadiki/pywhispercpp

# CoreML support
WHISPER_COREML=1 pip install git+https://github.com/absadiki/pywhispercpp
```

### Apple Silicon Mac用户（M系列芯片）

如果您使用的是Apple Silicon芯片的Mac，可以利用MLX框架来获得更好的性能：

```bash
pip install -r requirements-mlx.txt
```

查看 [MLX设置指南](docs/MLX_SETUP.md) 获取更多详细信息。

## 使用方法

### 使用OpenAI Whisper和Transformers摘要 (默认)

```python
from podcast_transcribe.transcriber import PodcastTranscriber

transcriber = PodcastTranscriber(
    whisper_model_path="/path/to/whisper/model",
    hf_token="your_huggingface_token"
)

summary, full_transcript = transcriber.transcribe_and_summarize("path/to/podcast.mp3")
print(summary)
```

### 使用whisper.cpp (更快的速度)

```python
from podcast_transcribe.transcriber import PodcastTranscriber

transcriber = PodcastTranscriber(
    whisper_model_path="base",  # 可以是模型名称，将自动下载
    hf_token="your_huggingface_token",
    transcriber_type="whisper-cpp"
)

summary, full_transcript = transcriber.transcribe_and_summarize("path/to/podcast.mp3")
print(summary)
```

### 使用Qwen本地模型进行高质量摘要

```python
from podcast_transcribe.transcriber import PodcastTranscriber

transcriber = PodcastTranscriber(
    whisper_model_path="/path/to/whisper/model",
    hf_token="your_huggingface_token",
    summarization_model="Qwen/Qwen1.5-1.8B",  # 也可以是本地模型路径
    summarizer_type="qwen",
    summarizer_device="cuda",
    summarizer_load_in_4bit=True  # 启用4bit量化节省显存
)

# 使用Qwen特有的生成参数
summary, full_transcript = transcriber.transcribe_and_summarize(
    "path/to/podcast.mp3",
    temperature=0.7,
    top_p=0.9
)
print(summary)
```

## 命令行使用

### 使用OpenAI Whisper和Transformers摘要 (默认)

```bash
podcast-transcribe --audio_file path/to/podcast.mp3 --whisper_model_path /path/to/model --hf_token your_token
```

### 使用whisper.cpp

```bash
podcast-transcribe --audio_file path/to/podcast.mp3 --whisper_model_path base --hf_token your_token --transcriber_type whisper-cpp
```

### 使用Qwen本地模型进行高质量摘要

```bash
podcast-transcribe \
  --audio_file path/to/podcast.mp3 \
  --whisper_model_path /path/to/model \
  --hf_token your_token \
  --summarizer_type qwen \
  --summarization_model Qwen/Qwen1.5-1.8B \
  --temperature 0.7 \
  --top_p 0.9
```

## 示例

查看`examples`目录的示例脚本：

- `summarize_with_qwen.py`: 使用Qwen本地模型进行播客内容摘要的示例
- `summarize_transcript_with_qwen.py`: 直接从已有转录文件生成摘要的示例
- `batch_summarize_transcripts.py`: 批量处理output目录下所有转录文件并生成摘要

### 直接从转录文件生成摘要

如果您已经有了转录文件，可以直接调用Qwen模型生成摘要：

```bash
# 从单个转录文件生成摘要
python examples/summarize_transcript_with_qwen.py \
  --transcript_file output/podcast_20240601_transcript.txt \
  --output_file output/podcast_20240601_summary.txt \
  --qwen_model_path Qwen/Qwen1.5-1.8B \
  --temperature 0.7

# 批量处理所有转录文件
python examples/batch_summarize_transcripts.py \
  --input_dir output \
  --pattern "*transcript*.txt" \
  --output_dir output/summaries \
  --qwen_model_path Qwen/Qwen1.5-1.8B
```
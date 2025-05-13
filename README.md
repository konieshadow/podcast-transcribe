# Podcast Transcriber

一个用于转录并总结播客音频内容的工具。支持使用OpenAI Whisper或whisper.cpp进行语音识别，pyannote进行说话人区分，以及Transformers进行文本摘要。

## 功能

- 将播客音频转换为文本
- 支持两种转录引擎：OpenAI Whisper和更快的whisper.cpp
- 区分不同说话人
- 为转录内容生成摘要

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

## 使用方法

### 使用OpenAI Whisper (默认)

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

## 命令行使用

### 使用OpenAI Whisper (默认)

```bash
podcast-transcribe --audio_file path/to/podcast.mp3 --whisper_model_path /path/to/model --hf_token your_token
```

### 使用whisper.cpp

```bash
podcast-transcribe --audio_file path/to/podcast.mp3 --whisper_model_path base --hf_token your_token --transcriber_type whisper-cpp
```
# Podcast Transcriber

一个用于转录并总结播客音频内容的工具。使用Whisper进行语音识别，pyannote进行说话人区分，以及Transformers进行文本摘要。

## 功能

- 将播客音频转换为文本
- 区分不同说话人
- 为转录内容生成摘要

## 使用方法

```python
from podcast_transcribe.transcriber import PodcastTranscriber

transcriber = PodcastTranscriber(
    whisper_model_path="/path/to/whisper/model",
    hf_token="your_huggingface_token"
)

summary, full_transcript = transcriber.transcribe_and_summarize("path/to/podcast.mp3")
print(summary)
```

## 命令行使用

```bash
podcast-transcribe --audio_file path/to/podcast.mp3 --whisper_model_path /path/to/model --hf_token your_token
```
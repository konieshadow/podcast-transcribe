"""
播客转录和摘要的主类
"""

import os
import logging
from typing import Tuple, Dict, Any, List, Optional, Literal

from .utils.audio import load_audio, extract_audio_segment
from .utils.diarization import SpeakerDiarizer
from .utils.transcription import AbstractTranscriber, WhisperTranscriber, WhisperCppTranscriber
from .utils.summarization import TextSummarizer, SummarizerType


# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 转录器类型
TranscriberType = Literal["openai-whisper", "whisper-cpp"]


class PodcastTranscriber:
    """用于转录和摘要播客的主类"""
    
    def __init__(
        self,
        whisper_model_path: str,
        hf_token: str,
        diarization_model: str = "pyannote/speaker-diarization-3.1",
        summarization_model: str = "facebook/bart-large-cnn",
        sample_rate: int = 16000,
        transcriber_type: TranscriberType = "openai-whisper",
        summarizer_type: SummarizerType = "transformers",
        summarizer_device: str = "cuda",
        summarizer_load_in_4bit: bool = True
    ):
        """
        初始化播客转录器
        
        参数:
            whisper_model_path: Whisper模型路径
            hf_token: Hugging Face API 令牌
            diarization_model: 说话人分割模型名称
            summarization_model: 摘要模型名称或路径
            sample_rate: 音频采样率（Hz）
            transcriber_type: 转录器类型，可选 "openai-whisper" 或 "whisper-cpp"
            summarizer_type: 摘要器类型，可选 "transformers" 或 "qwen"
            summarizer_device: 摘要器推理设备，"cuda"或"cpu"
            summarizer_load_in_4bit: 是否使用4bit量化加载摘要模型（仅对Qwen有效）
        """
        self.whisper_model_path = whisper_model_path
        self.hf_token = hf_token
        self.diarization_model = diarization_model
        self.summarization_model = summarization_model
        self.sample_rate = sample_rate
        self.transcriber_type = transcriber_type
        self.summarizer_type = summarizer_type
        self.summarizer_device = summarizer_device
        self.summarizer_load_in_4bit = summarizer_load_in_4bit
        
        # 初始化组件
        self._diarizer = None
        self._transcriber = None
        self._summarizer = None
        
        logger.info(f"播客转录器初始化完成，使用转录器类型: {transcriber_type}，摘要器类型: {summarizer_type}")
    
    @property
    def diarizer(self) -> SpeakerDiarizer:
        """懒加载说话人分割器"""
        if self._diarizer is None:
            logger.info(f"初始化说话人分割器: {self.diarization_model}")
            self._diarizer = SpeakerDiarizer(
                hf_token=self.hf_token,
                model_name=self.diarization_model
            )
        return self._diarizer
    
    @property
    def transcriber(self) -> AbstractTranscriber:
        """懒加载语音转录器"""
        if self._transcriber is None:
            logger.info(f"初始化语音转录器: {self.whisper_model_path}, 类型: {self.transcriber_type}")
            
            if self.transcriber_type == "whisper-cpp":
                self._transcriber = WhisperCppTranscriber(
                    model_path=self.whisper_model_path
                )
            else:  # 默认使用 openai-whisper
                self._transcriber = WhisperTranscriber(
                    model_path=self.whisper_model_path
                )
        return self._transcriber
    
    @property
    def summarizer(self) -> TextSummarizer:
        """懒加载文本摘要器"""
        if self._summarizer is None:
            logger.info(f"初始化文本摘要器: {self.summarization_model}, 类型: {self.summarizer_type}")
            self._summarizer = TextSummarizer(
                summarizer_type=self.summarizer_type,
                model_name_or_path=self.summarization_model,
                device=self.summarizer_device,
                load_in_4bit=self.summarizer_load_in_4bit
            )
        return self._summarizer
    
    def transcribe_and_summarize(
        self, 
        audio_file: str,
        max_summary_length: int = 150,
        min_summary_length: int = 30,
        output_transcript_file: Optional[str] = None,
        output_summary_file: Optional[str] = None,
        **summary_kwargs
    ) -> Tuple[str, str]:
        """
        转录和摘要播客音频文件
        
        参数:
            audio_file: 音频文件路径
            max_summary_length: 最大摘要长度
            min_summary_length: 最小摘要长度
            output_transcript_file: 输出转录文本的文件路径（如果提供）
            output_summary_file: 输出摘要文本的文件路径（如果提供）
            **summary_kwargs: 传递给摘要器的额外参数
            
        返回:
            摘要文本和完整转录文本的元组
        """
        logger.info(f"开始处理音频文件: {audio_file}")
        
        # 加载音频
        logger.info("加载音频并转换格式...")
        audio, waveform = load_audio(audio_file, target_sample_rate=self.sample_rate)
        
        # 执行说话人分割
        logger.info("进行说话人分割...")
        diarization_result = self.diarizer.diarize(waveform, sample_rate=self.sample_rate)
        
        # 初始化总转录文本
        total_transcription = ""
        
        # 转录每个说话人的音频段
        logger.info("转录各个说话人的音频段...")
        for speaker, segment in self.diarizer.get_speaker_segments(diarization_result):
            start_s = segment.start
            end_s = segment.end
            start_ms = int(start_s * 1000)
            end_ms = int(end_s * 1000)
            
            logger.debug(f"处理 {start_s:.2f}s - {end_s:.2f}s, 说话人: {speaker}")
            
            # 提取音频段
            audio_segment = extract_audio_segment(audio, start_ms, end_ms)
            
            # 转录音频段
            transcription_result = self.transcriber.transcribe(audio_segment)
            
            # 提取文本并添加到总转录
            for segment_info in self.transcriber.extract_text(transcription_result):
                text = segment_info['text']
                logger.debug(f"转录 start: {start_s:.2f}s, end: {end_s:.2f}s, 说话人 {speaker}: {text}")
                total_transcription += f"说话人 {speaker}: {text}\n"
        
        # 生成摘要
        logger.info("生成摘要...")
        summary = self.summarizer.summarize(
            text=total_transcription,
            max_length=max_summary_length,
            min_length=min_summary_length,
            **summary_kwargs
        )
        
        # 如果提供了输出文件路径，写入文件
        if output_transcript_file:
            logger.info(f"保存转录文本到: {output_transcript_file}")
            with open(output_transcript_file, 'w', encoding='utf-8') as f:
                f.write(total_transcription)
                
        if output_summary_file:
            logger.info(f"保存摘要文本到: {output_summary_file}")
            with open(output_summary_file, 'w', encoding='utf-8') as f:
                f.write(summary)
        
        logger.info("处理完成")
        return summary, total_transcription 
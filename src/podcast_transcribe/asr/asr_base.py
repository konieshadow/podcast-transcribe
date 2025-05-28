"""
语音识别模块基类
"""

import os
import numpy as np
from pydub import AudioSegment
from typing import Dict, List, Union, Optional, Tuple
# from dataclasses import dataclass # dataclass is now imported from schemas if needed or already there
import logging

from ..schemas import TranscriptionResult # Added import

# 配置日志
logger = logging.getLogger("asr")


class BaseTranscriber:
    """统一的语音识别基类，支持MLX和Transformers等多种框架"""
    
    def __init__(
        self, 
        model_name: str,
        device: str = None,
    ):
        """
        初始化转录器
        
        参数:
            model_name: 模型名称
            device: 推理设备，'cpu'或'cuda'，对于MLX框架此参数可忽略
        """
        self.model_name = model_name
        self.device = device
        self.pipeline = None  # 用于Transformers
        self.model = None     # 用于MLX等其他框架
                
        logger.info(f"初始化转录器，模型: {model_name}" + (f"，设备: {device}" if device else ""))
        
        # 子类需要实现_load_model方法
        self._load_model()
        
    def _load_model(self):
        """
        加载模型（需要在子类中实现）
        """
        raise NotImplementedError("子类必须实现_load_model方法")
    
    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        """
        转录音频，针对distil-whisper模型取消分块处理，直接处理整个音频。

        参数:
            audio: 要转录的AudioSegment对象

        返回:
            TranscriptionResult对象，包含转录结果
        """
        logger.info(f"开始转录 {len(audio)/1000:.2f} 秒的音频") # 移除了模型名称，因为基类不知道具体模型

        # 直接处理整个音频，不进行分块
        processed_audio = self._prepare_audio(audio)
        samples = np.array(processed_audio.get_array_of_samples(), dtype=np.float32) / 32768.0

        try:
            model_result = self._perform_transcription(samples)
            text = self._get_text_from_result(model_result)
            segments = self._convert_segments(model_result)
            language = self._detect_language(text)

            logger.info(f"转录完成，语言: {language}，文本长度: {len(text)}，分段数: {len(segments)}")
            return TranscriptionResult(text=text, segments=segments, language=language)
        except Exception as e:
            logger.error(f"转录失败: {str(e)}", exc_info=True)
            raise RuntimeError(f"转录失败: {str(e)}")

    def _get_text_from_result(self, result):
        """
        从结果中获取文本

        参数:
            result: 模型的转录结果

        返回:
            转录的文本
        """
        return result.get("text", "")

    def _perform_transcription(self, audio_data):
        """执行转录的抽象方法，由子类实现"""
        raise NotImplementedError("子类必须实现_perform_transcription方法")

    def _convert_segments(self, result) -> List[Dict[str, Union[float, str]]]:
        """将模型结果转换为分段的抽象方法，由子类实现"""
        raise NotImplementedError("子类必须实现_convert_segments方法")
    
    def _prepare_audio(self, audio: AudioSegment) -> AudioSegment:
        """
        准备音频数据
        
        参数:
            audio: 输入的AudioSegment对象
            
        返回:
            处理后的AudioSegment对象
        """
        logger.debug(f"准备音频数据: 时长={len(audio)/1000:.2f}秒, 采样率={audio.frame_rate}Hz, 声道数={audio.channels}")
        
        # 确保采样率为16kHz
        if audio.frame_rate != 16000:
            logger.debug(f"重采样音频从 {audio.frame_rate}Hz 到 16000Hz")
            audio = audio.set_frame_rate(16000)
            
        # 确保是单声道
        if audio.channels > 1:
            logger.debug(f"将{audio.channels}声道音频转换为单声道")
            audio = audio.set_channels(1)
            
        logger.debug(f"音频处理完成")
        
        return audio
    
    def _detect_language(self, text: str) -> str:
        """
        简单的语言检测（基于经验规则）
        
        参数:
            text: 识别出的文本
            
        返回:
            检测到的语言代码
        """
        # 简单的规则检测，实际应用中应使用更准确的语言检测
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        chinese_ratio = chinese_chars / len(text) if text else 0
        logger.debug(f"语言检测: 中文字符比例 = {chinese_ratio:.2f}")
        
        if chinese_chars > len(text) * 0.3:
            return "zh"
        return "en"
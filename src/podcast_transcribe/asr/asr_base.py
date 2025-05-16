"""
基于MLX实现的语音识别模块基类
"""

import os
import numpy as np
from pydub import AudioSegment
from typing import Dict, List, Union, Optional, Tuple
from dataclasses import dataclass
import logging

# 配置日志
logger = logging.getLogger("asr")

@dataclass
class TranscriptionResult:
    """转录结果数据类"""
    text: str  # 转录的文本
    segments: List[Dict[str, Union[float, str]]]  # 包含时间戳的分段
    language: str  # 检测到的语言


class BaseMLXTranscriber:
    """MLX语音识别基类"""
    
    def __init__(
        self, 
        model_name: str,
    ):
        """
        初始化转录器
        
        参数:
            model_name: 模型名称
        """
        self.model_name = model_name
                
        logger.info(f"初始化转录器，模型: {model_name}")
        
        # 子类需要实现_load_model方法
        self._load_model()
        
    def _load_model(self):
        """
        加载模型（需要在子类中实现）
        """
        raise NotImplementedError("子类必须实现_load_model方法")
    
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
    
    def _convert_segments(self, model_result) -> List[Dict[str, Union[float, str]]]:
        """
        将模型的分段结果转换为所需格式（需要在子类中实现）
        
        参数:
            model_result: 模型返回的结果
            
        返回:
            转换后的分段列表
        """
        raise NotImplementedError("子类必须实现_convert_segments方法")
    
    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        """
        转录音频
        
        参数:
            audio: 要转录的AudioSegment对象
            
        返回:
            TranscriptionResult对象，包含转录结果
        """
        logger.info(f"开始转录 {len(audio)/1000:.2f} 秒的音频")
        
        # 准备音频输入
        processed_audio = self._prepare_audio(audio)
        
        try:
            # 将AudioSegment转换为numpy数组
            logger.debug("将AudioSegment转换为numpy数组")
            samples = np.array(processed_audio.get_array_of_samples(), dtype=np.float32) / 32768.0
            
            # 执行转录（子类需要实现具体逻辑）
            logger.debug("开始执行推理")
            result = self._perform_transcription(samples)
            
            # 获取完整文本
            text = result.text if hasattr(result, 'text') else self._get_text_from_result(result)
            logger.debug(f"获得原始转录文本: {text[:50]}..." if len(text) > 50 else text)
            
            # 转换分段结果
            segments = self._convert_segments(result)
            
            # 检测语言
            language = self._detect_language(text)
            
            logger.info(f"转录完成，语言: {language}，文本长度: {len(text)}，分段数: {len(segments)}")
            
            return TranscriptionResult(
                text=text,
                segments=segments,
                language=language
            )
            
        except Exception as e:
            logger.error(f"转录失败: {str(e)}", exc_info=True)
            raise RuntimeError(f"转录失败: {str(e)}")
    
    def _perform_transcription(self, audio_data):
        """
        执行转录（需要在子类中实现）
        
        参数:
            audio_data: 音频数据（numpy数组）
            
        返回:
            模型的转录结果
        """
        raise NotImplementedError("子类必须实现_perform_transcription方法")
    
    def _get_text_from_result(self, result):
        """
        从结果中获取文本（需要在子类中实现）
        
        参数:
            result: 模型的转录结果
            
        返回:
            转录的文本
        """
        raise NotImplementedError("子类必须实现_get_text_from_result方法") 
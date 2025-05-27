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
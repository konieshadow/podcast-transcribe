"""
基于MLX实现的语音识别模块，使用distil-whisper-large-v3模型
"""

import os
from pydub import AudioSegment
from typing import Dict, List, Union
import logging
import numpy as np
import mlx_whisper

# 导入基类
from .asr_base import BaseTranscriber, TranscriptionResult

# 配置日志
logger = logging.getLogger("asr")


class MLXDistilWhisperTranscriber(BaseTranscriber):
    """使用MLX加载和运行distil-whisper-large-v3模型的转录器"""
    
    def __init__(
        self, 
        model_name: str = "mlx-community/distil-whisper-large-v3",
    ):
        """
        初始化转录器
        
        参数:
            model_name: 模型名称
        """
        super().__init__(model_name=model_name)
        
    def _load_model(self):
        """加载Distil Whisper模型"""
        try:
            # 懒加载mlx-whisper
            try:
                import mlx_whisper
            except ImportError:
                raise ImportError("请先安装mlx-whisper库: pip install mlx-whisper")
                
            logger.info(f"开始加载模型 {self.model_name}")
            self.model = mlx_whisper.load_models.load_model(self.model_name)
            logger.info(f"模型加载成功")
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}", exc_info=True)
            raise RuntimeError(f"加载模型失败: {str(e)}")
    
    def _convert_segments(self, result) -> List[Dict[str, Union[float, str]]]:
        """
        将模型的分段结果转换为所需格式
        
        参数:
            result: 模型返回的结果
            
        返回:
            转换后的分段列表
        """
        segments = []
        
        for segment in result.get("segments", []):
            segments.append({
                "start": segment.get("start", 0.0),
                "end": segment.get("end", 0.0),
                "text": segment.get("text", "").strip()
            })
        
        return segments
    
    def _perform_transcription(self, audio_data):
        """
        执行转录
        
        参数:
            audio_data: 音频数据（numpy数组）
            
        返回:
            模型的转录结果
        """
        return mlx_whisper.transcribe(audio_data, path_or_hf_repo=self.model_name)
    
    def _get_text_from_result(self, result):
        """
        从结果中获取文本
        
        参数:
            result: 模型的转录结果
            
        返回:
            转录的文本
        """
        return result.get("text", "")


def transcribe_audio(
    audio_segment: AudioSegment,
    model_name: str = "mlx-community/distil-whisper-large-v3",
) -> TranscriptionResult:
    """
    使用MLX和distil-whisper-large-v3模型转录音频
    
    参数:
        audio_segment: 输入的AudioSegment对象
        model_name: 使用的模型名称
        
    返回:
        TranscriptionResult对象，包含转录的文本、分段和语言
    """
    logger.info(f"调用transcribe_audio函数，音频长度: {len(audio_segment)/1000:.2f}秒")
    transcriber = MLXDistilWhisperTranscriber(model_name=model_name)
    return transcriber.transcribe(audio_segment)

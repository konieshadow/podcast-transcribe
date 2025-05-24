"""
基于MLX实现的语音识别模块，使用parakeet-tdt模型
"""

import os
from pydub import AudioSegment
from typing import Dict, List, Union
import logging
import tempfile
import numpy as np
import soundfile as sf

# 导入基类
from .asr_base import BaseTranscriber, TranscriptionResult

# 配置日志
logger = logging.getLogger("asr")


class MLXParakeetTranscriber(BaseTranscriber):
    """使用MLX加载和运行parakeet-tdt-0.6b-v2模型的转录器"""
    
    def __init__(
        self, 
        model_name: str = "mlx-community/parakeet-tdt-0.6b-v2",
    ):
        """
        初始化转录器
        
        参数:
            model_name: 模型名称
        """
        super().__init__(model_name=model_name)
    
    def _load_model(self):
        """加载Parakeet模型"""
        try:
            # 懒加载parakeet_mlx
            try:
                from parakeet_mlx import from_pretrained
            except ImportError:
                raise ImportError("请先安装parakeet-mlx库: pip install parakeet-mlx")
                
            logger.info(f"开始加载模型 {self.model_name}")
            self.model = from_pretrained(self.model_name)
            logger.info(f"模型加载成功")
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}", exc_info=True)
            raise RuntimeError(f"加载模型失败: {str(e)}")
    
    def _convert_segments(self, aligned_result) -> List[Dict[str, Union[float, str]]]:
        """
        将模型的分段结果转换为所需格式
        
        参数:
            aligned_result: 模型返回的分段结果
            
        返回:
            转换后的分段列表
        """
        segments = []
        
        for sentence in aligned_result.sentences:
            segments.append({
                "start": sentence.start,
                "end": sentence.end,
                "text": sentence.text
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
        # 由于parakeet-mlx可能不直接支持numpy数组输入
        # 创建临时文件并写入音频数据
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
            # 确保数据在[-1, 1]范围内
            if audio_data.max() > 1.0 or audio_data.min() < -1.0:
                audio_data = np.clip(audio_data, -1.0, 1.0)
            
            # 写入临时文件
            sf.write(temp_file.name, audio_data, 16000, 'PCM_16')
            
            # 使用临时文件进行转录
            result = self.model.transcribe(temp_file.name)
        
        return result
    
    def _get_text_from_result(self, result):
        """
        从结果中获取文本
        
        参数:
            result: 模型的转录结果
            
        返回:
            转录的文本
        """
        return result.text


def transcribe_audio(
    audio_segment: AudioSegment,
    model_name: str = "mlx-community/parakeet-tdt-0.6b-v2",
) -> TranscriptionResult:
    """
    使用MLX和parakeet-tdt模型转录音频
    
    参数:
        audio_segment: 输入的AudioSegment对象
        model_name: 使用的模型名称
        
    返回:
        TranscriptionResult对象，包含转录的文本、分段和语言
    """
    logger.info(f"调用transcribe_audio函数，音频长度: {len(audio_segment)/1000:.2f}秒")
    transcriber = MLXParakeetTranscriber(model_name=model_name)
    return transcriber.transcribe(audio_segment)

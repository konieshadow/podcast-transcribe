"""
基于Transformers实现的语音识别模块，使用distil-whisper-large-v3.5模型
"""

import os
from pydub import AudioSegment
from typing import Dict, List, Union
import logging
import numpy as np

# 导入基类
from .asr_base import BaseTranscriber, TranscriptionResult

# 配置日志
logger = logging.getLogger("asr")


class TransformersDistilWhisperTranscriber(BaseTranscriber):
    """使用Transformers加载和运行distil-whisper-large-v3.5模型的转录器"""
    
    def __init__(
        self, 
        model_name: str = "distil-whisper/distil-large-v3.5",
        device: str = "cpu",
    ):
        """
        初始化转录器
        
        参数:
            model_name: 模型名称
            device: 推理设备，'cpu'或'cuda'
        """
        super().__init__(model_name=model_name, device=device)
        
    def _load_model(self):
        """加载Distil Whisper模型"""
        try:
            # 懒加载transformers
            try:
                from transformers import pipeline
            except ImportError:
                raise ImportError("请先安装transformers库: pip install transformers")
                
            logger.info(f"开始加载模型 {self.model_name}")
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                return_timestamps=True,
                chunk_length_s=30,
                batch_size=16,
            )
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
        
        # transformers pipeline 的结果格式
        if "chunks" in result:
            for chunk in result["chunks"]:
                segments.append({
                    "start": chunk["timestamp"][0] if chunk["timestamp"][0] is not None else 0.0,
                    "end": chunk["timestamp"][1] if chunk["timestamp"][1] is not None else 0.0,
                    "text": chunk["text"].strip()
                })
        else:
            # 如果没有分段信息，创建一个单一分段
            segments.append({
                "start": 0.0,
                "end": 0.0,  # 无法确定结束时间
                "text": result.get("text", "").strip()
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
        # transformers pipeline 接受numpy数组作为输入
        # 音频数据已经在_prepare_audio中确保是16kHz采样率
        return self.pipeline(audio_data)
    
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
    model_name: str = "distil-whisper/distil-large-v3.5",
    device: str = "cpu",
) -> TranscriptionResult:
    """
    使用Transformers和distil-whisper-large-v3.5模型转录音频
    
    参数:
        audio_segment: 输入的AudioSegment对象
        model_name: 使用的模型名称
        device: 推理设备，'cpu'或'cuda'
        
    返回:
        TranscriptionResult对象，包含转录的文本、分段和语言
    """
    logger.info(f"调用transcribe_audio函数，音频长度: {len(audio_segment)/1000:.2f}秒")
    transcriber = TransformersDistilWhisperTranscriber(model_name=model_name, device=device)
    return transcriber.transcribe(audio_segment)

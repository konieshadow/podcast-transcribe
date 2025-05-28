"""
基于Transformers实现的语音识别模块，使用distil-whisper模型
"""

import os
from pydub import AudioSegment
from typing import Dict, List, Union, Literal
import logging
import numpy as np

# 导入基类
from .asr_base import BaseTranscriber, TranscriptionResult

# 配置日志
logger = logging.getLogger("asr")


class TransformersDistilWhisperTranscriber(BaseTranscriber):
    """使用Transformers加载和运行distil-whisper模型的转录器"""
    
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
        """加载Distil Whisper Transformers模型"""
        try:
            # 懒加载transformers
            try:
                from transformers import pipeline
            except ImportError:
                raise ImportError("请先安装transformers库: pip install transformers")
                
            logger.info(f"开始加载模型 {self.model_name} 设备: {self.device}")

            pipeline_device_arg = None
            if self.device == "cuda":
                pipeline_device_arg = 0  # 使用第一个 CUDA 设备
            elif self.device == "mps":
                pipeline_device_arg = "mps"  # 使用 MPS 设备
            elif self.device == "cpu":
                pipeline_device_arg = -1 # 使用 CPU
            else:
                # 对于其他未明确支持的 device 字符串，记录警告并默认使用 CPU
                logger.warning(f"不支持的设备字符串 '{self.device}'，将默认使用 CPU。")
                pipeline_device_arg = -1
            
            # 导入必要的模块来配置模型
            import warnings
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
            
            # 抑制特定的警告
            warnings.filterwarnings("ignore", message="The input name `inputs` is deprecated")
            warnings.filterwarnings("ignore", message="You have passed task=transcribe")
            warnings.filterwarnings("ignore", message="The attention mask is not set")
            
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=pipeline_device_arg,
                return_timestamps=True,
                chunk_length_s=30,      # 使用30秒的块长度
                stride_length_s=5,      # 块之间5秒的重叠
                batch_size=32,           # 顺序处理
                # 添加以下参数来减少警告
                generate_kwargs={
                    "task": "transcribe",
                    "language": None,  # 自动检测语言
                    "forced_decoder_ids": None,  # 避免冲突
                }
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
        
        # 确保音频数据格式正确
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # 使用正确的参数名称调用pipeline
        try:
            result = self.pipeline(
                audio_data,
                generate_kwargs={
                    "task": "transcribe",
                    "language": None,  # 自动检测语言
                    "forced_decoder_ids": None,  # 避免冲突
                }
            )
            return result
        except Exception as e:
            logger.warning(f"使用新参数格式失败，尝试使用默认参数: {str(e)}")
            # 如果新格式失败，回退到简单调用
            return self.pipeline(audio_data)

# 统一的接口函数
def transcribe_audio(
    audio_segment: AudioSegment,
    model_name: str = None,
    device: str = "cpu",
) -> TranscriptionResult:
    """
    使用Distil Whisper模型转录音频 (Transformers后端)
    
    参数:
        audio_segment: 输入的AudioSegment对象
        model_name: 使用的模型名称，如果不指定则使用默认模型
        device: 推理设备，'cpu'或'cuda'
        
    返回:
        TranscriptionResult对象，包含转录的文本、分段和语言
    """
    logger.info(f"调用 transcribe_audio 函数 (Transformers后端)，音频长度: {len(audio_segment)/1000:.2f}秒，设备: {device}")
    
    default_model = "distil-whisper/distil-large-v3.5"
    model = model_name or default_model
    transcriber = TransformersDistilWhisperTranscriber(model_name=model, device=device)
    
    return transcriber.transcribe(audio_segment)

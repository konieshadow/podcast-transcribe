"""
基于pyannote/speaker-diarization-3.1模型实现的说话人分离模块
"""

import os
import numpy as np
from pydub import AudioSegment
from typing import Any, Dict, List, Mapping, Text, Union, Optional, Tuple
import logging
import torch

from .diarizer_base import BaseDiarizer
from ..schemas import DiarizationResult

# 配置日志
logger = logging.getLogger("diarization")

class PyannoteTranscriber(BaseDiarizer):
    """使用pyannote/speaker-diarization-3.1模型进行说话人分离"""
    
    def __init__(
        self, 
        model_name: str = "pyannote/speaker-diarization-3.1",
        token: Optional[str] = None,
        device: str = "cpu",
        segmentation_batch_size: int = 32,
    ):
        """
        初始化说话人分离器
        
        参数:
            model_name: 模型名称
            token: Hugging Face令牌，用于访问模型
            device: 推理设备，'cpu'或'cuda'
            segmentation_batch_size: 分割批处理大小，默认为32
        """
        super().__init__(model_name, token, device, segmentation_batch_size)
        
        # 加载模型
        self._load_model()
        
    def _load_model(self):
        """加载pyannote模型"""
        try:
            # 懒加载pyannote.audio
            try:
                from pyannote.audio import Pipeline
            except ImportError:
                raise ImportError("请先安装pyannote.audio库: pip install pyannote.audio")
                
            logger.info(f"开始加载模型 {self.model_name}")
            self.pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=self.token
            )
            
            # 设置设备
            self.pipeline.to(torch.device(self.device))
            
            # 设置分割批处理大小
            if hasattr(self.pipeline, "segmentation_batch_size"):
                logger.info(f"设置分割批处理大小: {self.segmentation_batch_size}")
                self.pipeline.segmentation_batch_size = self.segmentation_batch_size
            
            logger.info(f"模型加载成功")
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}", exc_info=True)
            raise RuntimeError(f"加载模型失败: {str(e)}")
    

    
    def diarize(self, audio: AudioSegment) -> DiarizationResult:
        """
        对音频进行说话人分离
        
        参数:
            audio: 要处理的AudioSegment对象
            
        返回:
            DiarizationResult对象，包含分段结果和说话人数量
        """
        logger.info(f"开始处理 {len(audio)/1000:.2f} 秒的音频进行说话人分离")
        
        # 准备音频输入
        temp_audio_path = self._prepare_audio(audio)
        
        try:
            # 执行说话人分离
            logger.debug("开始执行说话人分离")
            from pyannote.audio.pipelines.utils.hook import ProgressHook

            # 自定义 ProgressHook 类
            class CustomProgressHook(ProgressHook):
                def __call__(
                    self,
                    step_name: Text,
                    step_artifact: Any,
                    file: Optional[Mapping] = None,
                    total: Optional[int] = None,
                    completed: Optional[int] = None,
                ):
                    if completed is not None:
                        logger.info(f"处理中 {step_name}: ({completed/total*100:.1f}%)")
                    else:
                        logger.info(f"已完成 {step_name}")

            with CustomProgressHook() as hook:
                diarization = self.pipeline(temp_audio_path, hook=hook)

            # 转换分段结果
            segments, num_speakers = self._convert_segments(diarization)
            
            logger.info(f"说话人分离完成，检测到 {num_speakers} 个说话人，生成 {len(segments)} 个分段")
            
            return DiarizationResult(
                segments=segments,
                num_speakers=num_speakers
            )
            
        except Exception as e:
            logger.error(f"说话人分离失败: {str(e)}", exc_info=True)
            raise RuntimeError(f"说话人分离失败: {str(e)}")
        finally:
            # 删除临时文件
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)


def diarize_audio(
    audio_segment: AudioSegment,
    model_name: str = "pyannote/speaker-diarization-3.1",
    token: Optional[str] = None,
    device: str = "cpu",
    segmentation_batch_size: int = 32,
) -> DiarizationResult:
    """
    使用pyannote模型对音频进行说话人分离
    
    参数:
        audio_segment: 输入的AudioSegment对象
        model_name: 使用的模型名称
        token: Hugging Face令牌
        device: 推理设备，'cpu'、'cuda'、'mps'
        segmentation_batch_size: 分割批处理大小，默认为32
        
    返回:
        DiarizationResult对象，包含分段和说话人数量
    """
    logger.info(f"调用diarize_audio函数，音频长度: {len(audio_segment)/1000:.2f}秒")
    transcriber = PyannoteTranscriber(model_name=model_name, token=token, device=device, segmentation_batch_size=segmentation_batch_size)
    return transcriber.diarize(audio_segment)

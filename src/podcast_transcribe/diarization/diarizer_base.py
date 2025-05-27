"""
说话人分离器基础类，包含可复用的方法
"""

import os
import logging
from abc import ABC, abstractmethod
from pydub import AudioSegment
from typing import Any, Dict, List, Union, Optional, Tuple

from ..schemas import DiarizationResult

# 配置日志
logger = logging.getLogger("diarization")


class BaseDiarizer(ABC):
    """说话人分离器基础类"""
    
    def __init__(
        self, 
        model_name: str,
        token: Optional[str] = None,
        device: str = "cpu",
        segmentation_batch_size: int = 32,
    ):
        """
        初始化说话人分离器基础参数
        
        参数:
            model_name: 模型名称
            token: Hugging Face令牌，用于访问模型
            device: 推理设备，'cpu'或'cuda'
            segmentation_batch_size: 分割批处理大小，默认为32
        """
        self.model_name = model_name
        self.device = device
        self.segmentation_batch_size = segmentation_batch_size
        
        logger.info(f"初始化说话人分离器，模型: {model_name}，设备: {device}，分割批处理大小: {segmentation_batch_size}")
    
    @abstractmethod
    def _load_model(self):
        """加载模型，子类需要实现"""
        pass
    
    def _prepare_audio(self, audio: AudioSegment) -> str:
        """
        准备音频数据，保存为临时文件
        
        参数:
            audio: 输入的AudioSegment对象
            
        返回:
            临时音频文件的路径
        """
        logger.debug(f"准备音频数据: 时长={len(audio)/1000:.2f}秒, 采样率={audio.frame_rate}Hz, 声道数={audio.channels}")
        
        # 确保采样率为16kHz (pyannote模型要求)
        if audio.frame_rate != 16000:
            logger.debug(f"重采样音频从 {audio.frame_rate}Hz 到 16000Hz")
            audio = audio.set_frame_rate(16000)
            
        # 确保是单声道
        if audio.channels > 1:
            logger.debug(f"将{audio.channels}声道音频转换为单声道")
            audio = audio.set_channels(1)
            
        # 保存为临时文件
        temp_audio_path = "_temp_audio_for_diarization.wav"
        audio.export(temp_audio_path, format="wav")
        
        logger.debug(f"音频处理完成，保存至: {temp_audio_path}")
        
        return temp_audio_path
    
    def _convert_segments(self, diarization) -> Tuple[List[Dict[str, Union[float, str, int]]], int]:
        """
        将pyannote的分段结果转换为所需格式
        
        参数:
            diarization: pyannote模型返回的分段结果
            
        返回:
            转换后的分段列表和说话人数量
        """
        segments = []
        speakers = set()
        
        # 遍历说话人分离结果
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
            speakers.add(speaker)
        
        # 按开始时间排序
        segments.sort(key=lambda x: x["start"])
        
        logger.debug(f"转换了 {len(segments)} 个分段，检测到 {len(speakers)} 个说话人")
        
        return segments, len(speakers)
    
    @abstractmethod
    def diarize(self, audio: AudioSegment) -> DiarizationResult:
        """
        对音频进行说话人分离，子类需要实现
        
        参数:
            audio: 要处理的AudioSegment对象
            
        返回:
            DiarizationResult对象，包含分段结果和说话人数量
        """
        pass 
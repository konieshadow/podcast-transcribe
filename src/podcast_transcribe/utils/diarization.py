"""
说话人分割工具模块
"""

import logging
import numpy as np
from typing import Dict, List, Any, Iterator, Tuple
from pyannote.audio import Pipeline
from pyannote.core import Segment
import torch


# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SpeakerDiarizer:
    """处理说话人分割的类"""

    def __init__(self, hf_token: str, model_name: str = "pyannote/speaker-diarization-3.1"):
        """
        初始化说话人分割器

        参数:
            hf_token: Hugging Face API 令牌
            model_name: pyannote 模型名称
        """
        self.hf_token = hf_token
        self.model_name = model_name
        self._pipeline = None
        # 检查 MPS/CUDA 可用性并设置设备
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        logger.debug(f"使用设备: {self.device}")
    
    @property
    def pipeline(self) -> Pipeline:
        """懒加载 pyannote 说话人分割管道"""
        if self._pipeline is None:
            try:
                self._pipeline = Pipeline.from_pretrained(
                    self.model_name, 
                    use_auth_token=self.hf_token
                )
                if self.device != "cpu":
                    self._pipeline.to(self.device)
            except Exception as e:
                raise RuntimeError(f"无法加载说话人分割模型: {str(e)}")
        return self._pipeline
    
    def diarize(self, waveform: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        执行说话人分割

        参数:
            waveform: 音频波形数据（numpy数组）
            sample_rate: 采样率，默认16kHz
            
        返回:
            说话人分割结果
        """
        try:
            # 将NumPy数组转换为torch张量，确保格式为(channel, time)
            if isinstance(waveform, np.ndarray):
                # 如果是单通道并且形状是一维的，需要添加通道维度
                if waveform.ndim == 1:
                    waveform = waveform[np.newaxis, :]
                waveform_tensor = torch.from_numpy(waveform)
            else:
                # 如果已经是张量，确保形状正确
                waveform_tensor = waveform
                
            return self.pipeline({"waveform": waveform_tensor, "sample_rate": sample_rate})
        except Exception as e:
            raise RuntimeError(f"说话人分割失败: {str(e)}")
    
    def get_speaker_segments(self, diarization_result) -> Iterator[Tuple[str, Segment]]:
        """
        获取说话人片段迭代器

        参数:
            diarization_result: 说话人分割结果
            
        返回:
            (说话人ID, 时间段)的迭代器
        """
        for turn in diarization_result.itertracks(yield_label=True):
            track, speaker = turn[0], turn[2]  # 正确解包返回的元组
            yield speaker, track 
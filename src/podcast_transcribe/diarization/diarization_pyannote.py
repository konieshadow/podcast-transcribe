"""
基于pyannote/speaker-diarization-3.1模型实现的说话人分离模块
"""

import os
import numpy as np
from pydub import AudioSegment
from typing import Dict, List, Union, Optional, Tuple
import logging
import torch

from ..schemas import DiarizationResult

# 配置日志
logger = logging.getLogger("diarization")

class PyannoteTranscriber:
    """使用pyannote/speaker-diarization-3.1模型进行说话人分离"""
    
    def __init__(
        self, 
        model_name: str = "pyannote/speaker-diarization-3.1",
        token: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        初始化说话人分离器
        
        参数:
            model_name: 模型名称
            token: Hugging Face令牌，用于访问模型
            device: 推理设备，'cpu'或'cuda'
        """
        self.model_name = model_name
        self.token = token or os.environ.get("HF_TOKEN")
        self.device = device
                
        logger.info(f"初始化说话人分离器，模型: {model_name}，设备: {device}")
        
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
                
            if not self.token:
                raise ValueError("需要提供Hugging Face令牌才能使用pyannote模型。请通过参数传入或设置HF_TOKEN环境变量。")
                
            logger.info(f"开始加载模型 {self.model_name}")
            self.pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=self.token
            )
            
            # 设置设备
            self.pipeline.to(torch.device(self.device))
            
            logger.info(f"模型加载成功")
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}", exc_info=True)
            raise RuntimeError(f"加载模型失败: {str(e)}")
    
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
            diarization = self.pipeline(temp_audio_path)
            
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
) -> DiarizationResult:
    """
    使用pyannote模型对音频进行说话人分离
    
    参数:
        audio_segment: 输入的AudioSegment对象
        model_name: 使用的模型名称
        token: Hugging Face令牌
        device: 推理设备，'cpu'、'cuda'、'mps'
        
    返回:
        DiarizationResult对象，包含分段和说话人数量
    """
    logger.info(f"调用diarize_audio函数，音频长度: {len(audio_segment)/1000:.2f}秒")
    transcriber = PyannoteTranscriber(model_name=model_name, token=token, device=device)
    return transcriber.diarize(audio_segment)

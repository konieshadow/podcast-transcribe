"""
音频处理工具模块
"""

import numpy as np
from io import BytesIO
from pydub import AudioSegment
from typing import Tuple, Dict, Any


def load_audio(audio_file: str, target_sample_rate: int = 16000, mono: bool = True) -> Tuple[AudioSegment, np.ndarray]:
    """
    加载音频文件并转换为目标采样率和通道数

    参数:
        audio_file: 音频文件路径
        target_sample_rate: 目标采样率，默认16kHz
        mono: 是否转换为单声道，默认True

    返回:
        AudioSegment对象和对应的numpy数组
    """
    try:
        audio = AudioSegment.from_file(audio_file)
        
        # 转换为单声道（如果需要）
        if mono and audio.channels > 1:
            audio = audio.set_channels(1)
            
        # 转换采样率
        if audio.frame_rate != target_sample_rate:
            audio = audio.set_frame_rate(target_sample_rate)
            
        # 获取音频波形（用于pyannote）
        waveform = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        
        return audio, waveform
        
    except Exception as e:
        raise RuntimeError(f"无法加载音频文件: {str(e)}")


def extract_audio_segment(audio: AudioSegment, start_ms: int, end_ms: int) -> BytesIO:
    """
    从音频中提取指定时间段

    参数:
        audio: AudioSegment对象
        start_ms: 开始时间（毫秒）
        end_ms: 结束时间（毫秒）
        
    返回:
        包含音频段的BytesIO对象
    """
    try:
        sub_audio = audio[start_ms:end_ms]
        fp = BytesIO()
        sub_audio.export(fp, format="wav")
        fp.seek(0)
        return fp
    except Exception as e:
        raise RuntimeError(f"无法提取音频段: {str(e)}") 
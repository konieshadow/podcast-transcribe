"""
语音转录工具模块
"""

from abc import ABC, abstractmethod
from io import BytesIO
import logging
from typing import Dict, Any, List
import whisper
import torch
import numpy as np
import soundfile as sf


# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AbstractTranscriber(ABC):
    """语音转录的抽象基类"""
    
    @abstractmethod
    def transcribe(self, audio_data: BytesIO) -> Dict[str, Any]:
        """
        转录音频数据
        
        参数:
            audio_data: 音频数据（BytesIO对象）
            
        返回:
            转录结果
        """
        pass
    
    @abstractmethod
    def extract_text(self, transcription_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从转录结果中提取文本
        
        参数:
            transcription_result: 转录结果
            
        返回:
            包含文本、时间戳等信息的片段列表
        """
        pass


class WhisperTranscriber(AbstractTranscriber):
    """使用OpenAI Whisper进行语音转录的类"""

    def __init__(self, model_path: str):
        """
        初始化Whisper转录器
        
        参数:
            model_path: Whisper模型路径
        """
        self.model_path = model_path
        self._model = None
        # 检查 CUDA 可用性并设置设备
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        logger.debug(f"使用设备: {self.device}")
    
    @property
    def model(self) -> whisper.Whisper:
        """懒加载Whisper模型"""
        if self._model is None:
            try:
                self._model = whisper.load_model(self.model_path, device=self.device)
            except Exception as e:
                raise RuntimeError(f"无法加载Whisper模型: {str(e)}")
        return self._model
    
    def transcribe(self, audio_data: BytesIO) -> Dict[str, Any]:
        """
        转录音频数据
        
        参数:
            audio_data: 音频数据（BytesIO对象）
            
        返回:
            转录结果
        """
        try:
            # 将 BytesIO 对象转换为 numpy 数组
            audio_data.seek(0)  # 确保从头开始读取
            audio_array, sample_rate = sf.read(audio_data)
            
            # 如果音频是立体声，转换为单声道
            if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                audio_array = audio_array.mean(axis=1)
            
            # 确保数据类型为float32
            audio_array = audio_array.astype(np.float32)
                
            return self.model.transcribe(audio_array, fp16=False)
        except Exception as e:
            raise RuntimeError(f"转录失败: {str(e)}")
    
    def extract_text(self, transcription_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从转录结果中提取文本
        
        参数:
            transcription_result: 转录结果
            
        返回:
            包含文本、时间戳等信息的片段列表
        """
        if 'segments' not in transcription_result:
            return []
            
        return transcription_result['segments']


class WhisperCppTranscriber(AbstractTranscriber):
    """使用whisper.cpp进行语音转录的类"""
    
    def __init__(self, model_path: str):
        """
        初始化WhisperCpp转录器
        
        参数:
            model_path: 模型名称或本地模型路径
        """
        try:
            # 动态导入，因为这是一个新的依赖
            from pywhispercpp.model import Model
            self.model_path = model_path
            self._model = None
            logger.debug(f"初始化WhisperCpp转录器，模型: {model_path}")
        except ImportError:
            raise ImportError("请先安装pywhispercpp: pip install pywhispercpp")
    
    @property
    def model(self):
        """懒加载WhisperCpp模型"""
        if self._model is None:
            try:
                from pywhispercpp.model import Model
                # 初始化模型，关闭实时打印和进度，这些会由我们自己的日志系统处理
                self._model = Model(
                    self.model_path, 
                    print_realtime=False, 
                    print_progress=False
                )
            except Exception as e:
                raise RuntimeError(f"无法加载WhisperCpp模型: {str(e)}")
        return self._model
    
    def transcribe(self, audio_data: BytesIO) -> Dict[str, Any]:
        """
        转录音频数据
        
        参数:
            audio_data: 音频数据（BytesIO对象）
            
        返回:
            转录结果，格式与WhisperTranscriber兼容
        """
        try:
            # 将 BytesIO 对象直接转换为 numpy 数组
            audio_data.seek(0)  # 确保从头开始读取
            audio_array, sample_rate = sf.read(audio_data)
            
            # 如果音频是立体声，转换为单声道
            if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                audio_array = audio_array.mean(axis=1)
            
            # 确保数据类型为float32并进行归一化
            audio_array = audio_array.astype(np.float32)
   
            segments = self.model.transcribe(audio_array)
            
            # 构建与原始Whisper兼容的结果格式
            result = {
                "segments": []
            }
            
            # 将pywhispercpp的段转换为与OpenAI Whisper兼容的格式
            for i, segment in enumerate(segments):
                result["segments"].append({
                    "id": i,
                    "start": segment.t0,
                    "end": segment.t1,
                    "text": segment.text,
                    "tokens": [],  # whisper.cpp不提供token级别信息，保留空列表以兼容接口
                    "temperature": 0.0,  # 填充兼容字段
                    "avg_logprob": 0.0,  # 填充兼容字段
                    "compression_ratio": 0.0,  # 填充兼容字段
                    "no_speech_prob": 0.0  # 填充兼容字段
                })
            
            return result
        except Exception as e:
            raise RuntimeError(f"WhisperCpp转录失败: {str(e)}")
    
    def extract_text(self, transcription_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从转录结果中提取文本
        
        参数:
            transcription_result: 转录结果
            
        返回:
            包含文本、时间戳等信息的片段列表
        """
        if 'segments' not in transcription_result:
            return []
            
        return transcription_result['segments'] 
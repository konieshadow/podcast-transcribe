"""
语音转录工具模块
"""

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

class WhisperTranscriber:
    """使用Whisper进行语音转录的类"""

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
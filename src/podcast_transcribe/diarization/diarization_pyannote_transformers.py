"""
基于transformers库调用pyannote/speaker-diarization-3.1模型实现的说话人分离模块
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


class PyannoteTransformersTranscriber(BaseDiarizer):
    """使用transformers库调用pyannote/speaker-diarization-3.1模型进行说话人分离"""
    
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
        """使用transformers加载pyannote模型"""
        try:
            # 检查依赖库
            try:
                from transformers import pipeline
                from transformers.pipelines import AutomaticSpeechRecognitionPipeline
            except ImportError:
                raise ImportError("请先安装transformers库: pip install transformers")
                
            try:
                import librosa
            except ImportError:
                raise ImportError("请先安装librosa库: pip install librosa")
                
            if not self.token:
                raise ValueError("需要提供Hugging Face令牌才能使用pyannote模型。请通过参数传入或设置HF_TOKEN环境变量。")
                
            logger.info(f"开始使用transformers加载模型 {self.model_name}")
            
            # 使用transformers pipeline加载音频分类模型
            self.pipeline = pipeline(
                "audio-classification",
                model=self.model_name,
                use_auth_token=self.token,
                device=0 if self.device == "cuda" and torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            logger.info(f"transformers模型加载成功")
            
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}", exc_info=True)
            # 回退到pyannote.audio方式
            logger.info("尝试回退到pyannote.audio方式加载模型")
            try:
                from pyannote.audio import Pipeline
                
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
                
                logger.info(f"使用pyannote.audio加载模型成功")
                self._use_pyannote_audio = True
                
            except Exception as fallback_error:
                logger.error(f"回退加载也失败: {str(fallback_error)}", exc_info=True)
                raise RuntimeError(f"模型加载失败: {str(e)}，回退也失败: {str(fallback_error)}")
    
    def _process_with_transformers(self, audio_path: str):
        """使用transformers处理音频文件"""
        try:
            import librosa
            
            # 加载音频文件
            waveform, sample_rate = librosa.load(audio_path, sr=16000)
            
            # 注意：transformers的audio-classification pipeline通常用于分类任务
            # 对于说话人分离，我们需要使用专门的方法
            # 这里我们实际上还是需要使用pyannote的方法，因为transformers没有直接的说话人分离pipeline
            logger.warning("transformers库目前不直接支持说话人分离，回退到pyannote.audio")
            
            # 加载pyannote pipeline作为备选
            from pyannote.audio import Pipeline
            
            pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=self.token
            )
            
            pipeline.to(torch.device(self.device))
            
            return pipeline(audio_path)
            
        except Exception as e:
            logger.error(f"transformers处理失败: {str(e)}")
            raise
    
    def diarize(self, audio: AudioSegment) -> DiarizationResult:
        """
        对音频进行说话人分离
        
        参数:
            audio: 要处理的AudioSegment对象
            
        返回:
            DiarizationResult对象，包含分段结果和说话人数量
        """
        logger.info(f"开始使用transformers处理 {len(audio)/1000:.2f} 秒的音频进行说话人分离")
        
        # 准备音频输入
        temp_audio_path = self._prepare_audio(audio)
        
        try:
            # 执行说话人分离
            logger.debug("开始执行说话人分离")
            
            # 检查是否使用pyannote.audio作为回退
            if hasattr(self, '_use_pyannote_audio') and self._use_pyannote_audio:
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
            else:
                # 使用transformers方式（实际上还是会回退到pyannote）
                diarization = self._process_with_transformers(temp_audio_path)

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
    使用transformers调用pyannote模型对音频进行说话人分离
    
    参数:
        audio_segment: 输入的AudioSegment对象
        model_name: 使用的模型名称
        token: Hugging Face令牌
        device: 推理设备，'cpu'、'cuda'、'mps'
        segmentation_batch_size: 分割批处理大小，默认为32
        
    返回:
        DiarizationResult对象，包含分段和说话人数量
    """
    logger.info(f"调用transformers版本diarize_audio函数，音频长度: {len(audio_segment)/1000:.2f}秒")
    transcriber = PyannoteTransformersTranscriber(
        model_name=model_name, 
        token=token, 
        device=device, 
        segmentation_batch_size=segmentation_batch_size
    )
    return transcriber.diarize(audio_segment)

"""
ASR模型调用路由器
根据传递的provider参数调用不同的ASR实现，支持延迟加载
"""

import logging
from typing import Dict, Any, Optional, Callable
from pydub import AudioSegment
import spaces
from .asr_base import TranscriptionResult

# 配置日志
logger = logging.getLogger("asr")


class ASRRouter:
    """ASR模型调用路由器，支持多种ASR实现的统一调用"""
    
    def __init__(self):
        """初始化路由器"""
        self._loaded_modules = {}  # 用于缓存已加载的模块
        self._transcribers = {}    # 用于缓存已实例化的转录器
        
        # 定义支持的provider配置
        self._provider_configs = {
            "distil_whisper_transformers": {
                "module_path": ".asr_distil_whisper_transformers",
                "function_name": "transcribe_audio",
                "default_model": "distil-whisper/distil-large-v3.5",
                "supported_params": ["model_name", "device"],
                "description": "基于Transformers的Distil Whisper模型"
            }
        }
    
    def _lazy_load_module(self, provider: str):
        """
        获取指定provider的模块
        
        参数:
            provider: provider名称
            
        返回:
            对应的模块
        """
        if provider not in self._provider_configs:
            raise ValueError(f"不支持的provider: {provider}")
            
        if provider not in self._loaded_modules:
            module_path = self._provider_configs[provider]["module_path"]
            logger.info(f"获取模块: {module_path}")
            
            # 使用 importlib 动态导入模块
            import importlib
            module = importlib.import_module(module_path, package=__package__)
            
            self._loaded_modules[provider] = module
            logger.info(f"模块 {module_path} 获取成功")
        
        return self._loaded_modules[provider]
    
    def _get_transcribe_function(self, provider: str) -> Callable:
        """
        获取指定provider的转录函数
        
        参数:
            provider: provider名称
            
        返回:
            转录函数
        """
        module = self._lazy_load_module(provider)
        function_name = self._provider_configs[provider]["function_name"]
        
        if not hasattr(module, function_name):
            raise AttributeError(f"模块中未找到函数: {function_name}")
            
        return getattr(module, function_name)
    
    def _filter_params(self, provider: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        过滤参数，只保留指定provider支持的参数
        
        参数:
            provider: provider名称
            params: 原始参数字典
            
        返回:
            过滤后的参数字典
        """
        supported_params = self._provider_configs[provider]["supported_params"]
        filtered_params = {}
        
        for param in supported_params:
            if param in params:
                filtered_params[param] = params[param]
        
        # 如果没有指定model_name，使用默认模型
        if "model_name" not in filtered_params and "model_name" in supported_params:
            filtered_params["model_name"] = self._provider_configs[provider]["default_model"]
        
        # 对于 Transformers backend，如果 device 未指定，则默认为 cpu
        if provider == "distil_whisper_transformers" and "device" in supported_params and "device" not in filtered_params:
            filtered_params["device"] = "cpu"
        
        return filtered_params
    
    def transcribe(
        self,
        audio_segment: AudioSegment,
        provider: str,
        **kwargs
    ) -> TranscriptionResult:
        """
        统一的音频转录接口
        
        参数:
            audio_segment: 输入的AudioSegment对象
            provider: ASR提供者名称
            **kwargs: 其他参数，如model_name, device等
            
        返回:
            TranscriptionResult对象
        """
        logger.info(f"使用provider '{provider}' 进行音频转录，音频长度: {len(audio_segment)/1000:.2f}秒")
        
        if provider not in self._provider_configs:
            available_providers = list(self._provider_configs.keys())
            raise ValueError(f"不支持的provider: {provider}。支持的provider: {available_providers}")
        
        try:
            # 获取转录函数
            transcribe_func = self._get_transcribe_function(provider)
            
            # 过滤并准备参数
            filtered_kwargs = self._filter_params(provider, kwargs)
            
            logger.debug(f"调用 {provider} 转录函数，参数: {filtered_kwargs}")
            
            # 执行转录
            result = transcribe_func(audio_segment, **filtered_kwargs)
            
            logger.info(f"转录完成，文本长度: {len(result.text)}字符")
            return result
            
        except Exception as e:
            logger.error(f"使用provider '{provider}' 转录音频失败: {str(e)}", exc_info=True)
            raise RuntimeError(f"转录失败: {str(e)}")
    
    def get_available_providers(self) -> Dict[str, str]:
        """
        获取所有可用的provider及其描述
        
        返回:
            provider名称到描述的映射
        """
        return {
            provider: config["description"] 
            for provider, config in self._provider_configs.items()
        }
    
    def get_provider_info(self, provider: str) -> Dict[str, Any]:
        """
        获取指定provider的详细信息
        
        参数:
            provider: provider名称
            
        返回:
            provider的配置信息
        """
        if provider not in self._provider_configs:
            raise ValueError(f"不支持的provider: {provider}")
            
        return self._provider_configs[provider].copy()


# 创建全局路由器实例
_router = ASRRouter()

@spaces.GPU(duration=180)
def transcribe_audio(
    audio_segment: AudioSegment,
    provider: str = "distil_whisper_transformers",
    model_name: Optional[str] = None,
    device: str = "cpu",
    **kwargs
) -> TranscriptionResult:
    """
    统一的音频转录接口，通过路由器选择后端
    """
    # 准备参数
    params = kwargs.copy()
    if model_name is not None:
        params["model_name"] = model_name
    if device != "cpu": # 只有当 device 不是默认值才传递，或者根据需要传递所有支持的参数
        params["device"] = device
    
    return _router.transcribe(audio_segment, provider, **params)


def get_available_providers() -> Dict[str, str]:
    """
    获取所有可用的ASR提供者
    
    返回:
        provider名称到描述的映射
    """
    return _router.get_available_providers()


def get_provider_info(provider: str) -> Dict[str, Any]:
    """
    获取指定provider的详细信息
    
    参数:
        provider: provider名称
        
    返回:
        provider的配置信息
    """
    return _router.get_provider_info(provider)

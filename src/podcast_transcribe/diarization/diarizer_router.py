"""
说话人分离模型调用路由器
根据传递的provider参数调用不同的说话人分离实现，支持延迟加载
"""

import logging
from typing import Dict, Any, Optional, Callable
from pydub import AudioSegment
import spaces
from ..schemas import DiarizationResult
from . import diarization_pyannote_mlx
from . import diarization_pyannote_transformers

# 配置日志
logger = logging.getLogger("diarization")


class DiarizerRouter:
    """说话人分离模型调用路由器，支持多种实现的统一调用"""
    
    def __init__(self):
        """初始化路由器"""
        self._loaded_modules = {}  # 用于缓存已加载的模块
        self._diarizers = {}       # 用于缓存已实例化的分离器
        
        # 定义支持的provider配置
        self._provider_configs = {
            "pyannote_mlx": {
                "module_path": "diarization_pyannote_mlx",
                "function_name": "diarize_audio",
                "default_model": "pyannote/speaker-diarization-3.1",
                "supported_params": ["model_name", "token", "device", "segmentation_batch_size"],
                "description": "基于pyannote.audio的原生MLX实现"
            },
            "pyannote_transformers": {
                "module_path": "diarization_pyannote_transformers",
                "function_name": "diarize_audio",
                "default_model": "pyannote/speaker-diarization-3.1",
                "supported_params": ["model_name", "token", "device", "segmentation_batch_size"],
                "description": "基于transformers库调用pyannote模型"
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
            
            # 根据module_path返回对应的模块
            if module_path == "diarization_pyannote_mlx":
                module = diarization_pyannote_mlx
            elif module_path == "diarization_pyannote_transformers":
                module = diarization_pyannote_transformers
            else:
                raise ImportError(f"未找到模块: {module_path}")
            
            self._loaded_modules[provider] = module
            logger.info(f"模块 {module_path} 获取成功")
        
        return self._loaded_modules[provider]
    
    def _get_diarize_function(self, provider: str) -> Callable:
        """
        获取指定provider的说话人分离函数
        
        参数:
            provider: provider名称
            
        返回:
            说话人分离函数
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
        
        return filtered_params
    
    def diarize(
        self,
        audio_segment: AudioSegment,
        provider: str,
        **kwargs
    ) -> DiarizationResult:
        """
        统一的说话人分离接口
        
        参数:
            audio_segment: 输入的AudioSegment对象
            provider: 说话人分离提供者名称
            **kwargs: 其他参数，如model_name, token, device, segmentation_batch_size等
            
        返回:
            DiarizationResult对象
        """
        logger.info(f"使用provider '{provider}' 进行说话人分离，音频长度: {len(audio_segment)/1000:.2f}秒")
        
        if provider not in self._provider_configs:
            available_providers = list(self._provider_configs.keys())
            raise ValueError(f"不支持的provider: {provider}。支持的provider: {available_providers}")
        
        try:
            # 获取说话人分离函数
            diarize_func = self._get_diarize_function(provider)
            
            # 过滤并准备参数
            filtered_kwargs = self._filter_params(provider, kwargs)
            
            logger.debug(f"调用 {provider} 说话人分离函数，参数: {filtered_kwargs}")
            
            # 执行说话人分离
            result = diarize_func(audio_segment, **filtered_kwargs)
            
            logger.info(f"说话人分离完成，检测到 {result.num_speakers} 个说话人，生成 {len(result.segments)} 个分段")
            return result
            
        except Exception as e:
            logger.error(f"使用provider '{provider}' 进行说话人分离失败: {str(e)}", exc_info=True)
            raise RuntimeError(f"说话人分离失败: {str(e)}")
    
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
_router = DiarizerRouter()

@spaces.GPU(duration=180)
def diarize_audio(
    audio_segment: AudioSegment,
    provider: str = "pyannote_mlx",
    model_name: Optional[str] = None,
    token: Optional[str] = None,
    device: str = "cpu",
    segmentation_batch_size: int = 32,
    **kwargs
) -> DiarizationResult:
    """
    统一的音频说话人分离接口函数
    
    参数:
        audio_segment: 输入的AudioSegment对象
        provider: 说话人分离提供者，可选值：
            - "pyannote_mlx": 基于pyannote.audio的原生MLX实现
            - "pyannote_transformers": 基于transformers库调用pyannote模型
        model_name: 模型名称，如果不指定则使用默认模型
        token: Hugging Face令牌，用于访问模型
        device: 推理设备，'cpu'、'cuda'、'mps'
        segmentation_batch_size: 分割批处理大小，默认为32
        **kwargs: 其他参数
        
    返回:
        DiarizationResult对象，包含分段结果和说话人数量
        
    示例:
        # 使用默认pyannote MLX实现
        result = diarize_audio(audio_segment, provider="pyannote_mlx", token="your_hf_token")
        
        # 使用transformers实现
        result = diarize_audio(
            audio_segment, 
            provider="pyannote_transformers",
            token="your_hf_token"
        )
        
        # 使用GPU设备
        result = diarize_audio(
            audio_segment,
            provider="pyannote_mlx",
            token="your_hf_token",
            device="cuda"
        )
        
        # 自定义批处理大小
        result = diarize_audio(
            audio_segment,
            provider="pyannote_mlx", 
            token="your_hf_token",
            segmentation_batch_size=64
        )
    """
    # 准备参数
    params = kwargs.copy()
    if model_name is not None:
        params["model_name"] = model_name
    if token is not None:
        params["token"] = token
    if device != "cpu":
        params["device"] = device
    if segmentation_batch_size != 32:
        params["segmentation_batch_size"] = segmentation_batch_size
    
    return _router.diarize(audio_segment, provider, **params)


def get_available_providers() -> Dict[str, str]:
    """
    获取所有可用的说话人分离提供者
    
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

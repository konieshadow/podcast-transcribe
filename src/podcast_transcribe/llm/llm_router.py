"""
LLM模型调用路由器
根据传递的provider参数调用不同的LLM实现，支持延迟加载
"""

import logging
from typing import Dict, Any, Optional, List, Union
from .llm_base import BaseChatCompletion
from . import llm_gemma_mlx
from . import llm_gemma_transfomers
from . import llm_phi4_transfomers

# 配置日志
logger = logging.getLogger("llm")


class LLMRouter:
    """LLM模型调用路由器，支持多种实现的统一调用"""
    
    def __init__(self):
        """初始化路由器"""
        self._loaded_modules = {}  # 用于缓存已加载的模块
        self._llm_instances = {}   # 用于缓存已实例化的LLM实例
        
        # 定义支持的provider配置
        self._provider_configs = {
            "mlx": {
                "module_path": "llm_gemma_mlx",
                "class_name": "GemmaMLXChatCompletion",
                "default_model": "mlx-community/gemma-3-12b-it-4bit-DWQ",
                "supported_params": ["model_name"],
                "description": "基于MLX库的Gemma聊天完成实现"
            },
            "gemma-transformers": {
                "module_path": "llm_gemma_transfomers",
                "class_name": "GemmaTransformersChatCompletion",
                "default_model": "google/gemma-3-12b-it",
                "supported_params": [
                    "model_name", "use_4bit_quantization", "device_map", 
                    "device", "trust_remote_code"
                ],
                "description": "基于Transformers库的Gemma聊天完成实现"
            },
            "phi4-transformers": {
                "module_path": "llm_phi4_transfomers",
                "class_name": "Phi4TransformersChatCompletion",
                "default_model": "microsoft/Phi-4-mini-reasoning",
                "supported_params": [
                    "model_name", "use_4bit_quantization", "device_map", 
                    "device", "trust_remote_code", "enable_reasoning"
                ],
                "description": "基于Transformers库的Phi-4推理聊天完成实现"
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
            if module_path == "llm_gemma_mlx":
                module = llm_gemma_mlx
            elif module_path == "llm_gemma_transfomers":
                module = llm_gemma_transfomers
            elif module_path == "llm_phi4_transfomers":
                module = llm_phi4_transfomers
            else:
                raise ImportError(f"未找到模块: {module_path}")
            
            self._loaded_modules[provider] = module
            logger.info(f"模块 {module_path} 获取成功")
        
        return self._loaded_modules[provider]
    
    def _get_llm_class(self, provider: str):
        """
        获取指定provider的LLM类
        
        参数:
            provider: provider名称
            
        返回:
            LLM类
        """
        module = self._lazy_load_module(provider)
        class_name = self._provider_configs[provider]["class_name"]
        
        if not hasattr(module, class_name):
            raise AttributeError(f"模块中未找到类: {class_name}")
            
        return getattr(module, class_name)
    
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
    
    def _get_instance_key(self, provider: str, params: Dict[str, Any]) -> str:
        """
        生成LLM实例的缓存键
        
        参数:
            provider: provider名称
            params: 参数字典
            
        返回:
            实例缓存键
        """
        # 将参数转换为可哈希的字符串
        param_str = "_".join([f"{k}={v}" for k, v in sorted(params.items())])
        return f"{provider}_{param_str}"
    
    def _get_or_create_instance(self, provider: str, **kwargs) -> BaseChatCompletion:
        """
        获取或创建LLM实例（支持缓存复用）
        
        参数:
            provider: provider名称
            **kwargs: 构造函数参数
            
        返回:
            LLM实例
        """
        # 过滤并准备参数
        filtered_kwargs = self._filter_params(provider, kwargs)
        
        # 生成实例缓存键
        instance_key = self._get_instance_key(provider, filtered_kwargs)
        
        # 检查是否已有缓存实例
        if instance_key not in self._llm_instances:
            try:
                # 获取LLM类
                llm_class = self._get_llm_class(provider)
                
                logger.debug(f"创建 {provider} LLM实例，参数: {filtered_kwargs}")
                
                # 创建实例
                instance = llm_class(**filtered_kwargs)
                
                # 缓存实例
                self._llm_instances[instance_key] = instance
                
                logger.info(f"LLM实例创建成功: {provider} ({instance.model_name})")
                
            except Exception as e:
                logger.error(f"创建 {provider} LLM实例失败: {str(e)}", exc_info=True)
                raise RuntimeError(f"创建LLM实例失败: {str(e)}")
        
        return self._llm_instances[instance_key]
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        provider: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        统一的聊天完成接口
        
        参数:
            messages: 消息列表，每个消息包含role和content
            provider: LLM提供者名称
            temperature: 温度参数，控制生成的随机性
            max_tokens: 最大生成token数
            top_p: nucleus采样参数
            model: 可选的模型名称，如果提供则覆盖默认model_name
            **kwargs: 其他参数，如device、use_4bit_quantization等
            
        返回:
            聊天完成响应字典
        """
        logger.info(f"使用provider '{provider}' 进行聊天完成，消息数量: {len(messages)}")
        
        if provider not in self._provider_configs:
            available_providers = list(self._provider_configs.keys())
            raise ValueError(f"不支持的provider: {provider}。支持的provider: {available_providers}")
        
        try:
            # 如果提供了model参数，添加到kwargs中
            if model is not None:
                kwargs["model_name"] = model
            
            # 获取或创建LLM实例
            llm_instance = self._get_or_create_instance(provider, **kwargs)
            
            # 调用聊天完成
            result = llm_instance.create(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                model=model,
                **kwargs
            )
            
            logger.info(f"聊天完成成功，使用tokens: {result.get('usage', {}).get('total_tokens', 'unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"使用provider '{provider}' 进行聊天完成失败: {str(e)}", exc_info=True)
            raise RuntimeError(f"聊天完成失败: {str(e)}")
    
    def reasoning_completion(
        self,
        messages: List[Dict[str, str]],
        provider: str = "phi4-transformers",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        model: Optional[str] = None,
        extract_reasoning_steps: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        专门用于推理任务的聊天完成接口
        
        参数:
            messages: 消息列表，每个消息包含role和content
            provider: LLM提供者名称，默认使用phi4-transformers
            temperature: 温度参数（推理任务建议使用较低值）
            max_tokens: 最大生成token数
            top_p: nucleus采样参数
            model: 可选的模型名称
            extract_reasoning_steps: 是否提取推理步骤
            **kwargs: 其他参数
            
        返回:
            包含推理步骤的响应字典
        """
        logger.info(f"使用provider '{provider}' 进行推理完成，消息数量: {len(messages)}")
        
        # 确保使用支持推理的provider
        if provider not in ["phi4-transformers"]:
            logger.warning(f"Provider '{provider}' 可能不支持推理功能，建议使用 'phi4-transformers'")
        
        try:
            # 如果提供了model参数，添加到kwargs中
            if model is not None:
                kwargs["model_name"] = model
            
            # 获取或创建LLM实例
            llm_instance = self._get_or_create_instance(provider, **kwargs)
            
            # 检查实例是否支持推理完成
            if hasattr(llm_instance, 'reasoning_completion'):
                result = llm_instance.reasoning_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    extract_reasoning_steps=extract_reasoning_steps,
                    **kwargs
                )
            else:
                # 回退到普通聊天完成
                logger.warning(f"Provider '{provider}' 不支持推理完成，回退到普通聊天完成")
                result = llm_instance.create(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    model=model,
                    **kwargs
                )
            
            logger.info(f"推理完成成功，使用tokens: {result.get('usage', {}).get('total_tokens', 'unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"使用provider '{provider}' 进行推理完成失败: {str(e)}", exc_info=True)
            raise RuntimeError(f"推理完成失败: {str(e)}")
    
    def get_model_info(self, provider: str, **kwargs) -> Dict[str, Any]:
        """
        获取模型信息
        
        参数:
            provider: provider名称
            **kwargs: 构造函数参数
            
        返回:
            模型信息字典
        """
        try:
            llm_instance = self._get_or_create_instance(provider, **kwargs)
            return llm_instance.get_model_info()
        except Exception as e:
            logger.error(f"获取模型信息失败: {str(e)}")
            raise RuntimeError(f"获取模型信息失败: {str(e)}")
    
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
    
    def clear_cache(self):
        """清理缓存的实例"""
        # 清理每个实例的GPU缓存
        for instance in self._llm_instances.values():
            if hasattr(instance, 'clear_cache'):
                instance.clear_cache()
        
        # 清理实例缓存
        self._llm_instances.clear()
        logger.info("LLM实例缓存已清理")


# 创建全局路由器实例
_router = LLMRouter()


def chat_completion(
    messages: List[Dict[str, str]],
    provider: str = "mlx",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    top_p: float = 1.0,
    model: Optional[str] = None,
    device: Optional[str] = None,
    use_4bit_quantization: bool = False,
    device_map: Optional[str] = "auto",
    trust_remote_code: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    统一的聊天完成接口函数
    
    参数:
        messages: 消息列表，每个消息包含role和content字段
        provider: LLM提供者，可选值：
            - "mlx": 基于MLX库的Gemma聊天完成实现
            - "gemma-transformers": 基于Transformers库的Gemma聊天完成实现
            - "phi4-transformers": 基于Transformers库的Phi-4推理聊天完成实现
            - "transformers": 向后兼容别名，等同于gemma-transformers
        temperature: 温度参数，控制生成的随机性 (0.0-2.0)
        max_tokens: 最大生成token数
        top_p: nucleus采样参数 (0.0-1.0)
        model: 模型名称，如果不指定则使用默认模型
        device: 推理设备，'cpu'、'cuda'、'mps'（仅transformers provider支持）
        use_4bit_quantization: 是否使用4bit量化（仅transformers provider支持）
        device_map: 设备映射配置（仅transformers provider支持）
        trust_remote_code: 是否信任远程代码（仅transformers provider支持）
        **kwargs: 其他参数
        
    返回:
        聊天完成响应字典，包含生成的消息和使用统计
        
    示例:
        # 使用默认MLX实现
        response = chat_completion(
            messages=[{"role": "user", "content": "你好"}],
            provider="mlx"
        )
        
        # 使用Gemma transformers实现
        response = chat_completion(
            messages=[{"role": "user", "content": "你好"}],
            provider="gemma-transformers",
            model="google/gemma-3-12b-it",
            device="cuda",
            use_4bit_quantization=True
        )
        
        # 使用Phi-4推理实现
        response = chat_completion(
            messages=[{"role": "user", "content": "解这个数学题：2x + 5 = 15"}],
            provider="phi4-transformers",
            model="microsoft/Phi-4-mini-reasoning",
            device="cuda"
        )
        
        # 自定义参数
        response = chat_completion(
            messages=[
                {"role": "system", "content": "你是一个有用的助手"},
                {"role": "user", "content": "请介绍自己"}
            ],
            provider="mlx",
            temperature=0.8,
            max_tokens=1024
        )
    """
    # 准备参数
    params = kwargs.copy()
    if model is not None:
        params["model_name"] = model
    if device is not None:
        params["device"] = device
    if use_4bit_quantization:
        params["use_4bit_quantization"] = use_4bit_quantization
    if device_map != "auto":
        params["device_map"] = device_map
    if not trust_remote_code:
        params["trust_remote_code"] = trust_remote_code
    
    return _router.chat_completion(
        messages=messages,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        model=model,
        **params
    )


def reasoning_completion(
    messages: List[Dict[str, str]],
    provider: str = "phi4-transformers",
    temperature: float = 0.3,
    max_tokens: int = 2048,
    top_p: float = 0.9,
    model: Optional[str] = None,
    device: Optional[str] = None,
    use_4bit_quantization: bool = False,
    device_map: Optional[str] = "auto",
    trust_remote_code: bool = True,
    extract_reasoning_steps: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    专门用于推理任务的聊天完成接口函数
    
    参数:
        messages: 消息列表，每个消息包含role和content字段
        provider: LLM提供者，默认使用phi4-transformers
        temperature: 温度参数（推理任务建议使用较低值）
        max_tokens: 最大生成token数
        top_p: nucleus采样参数
        model: 模型名称，如果不指定则使用默认模型
        device: 推理设备
        use_4bit_quantization: 是否使用4bit量化
        device_map: 设备映射配置
        trust_remote_code: 是否信任远程代码
        extract_reasoning_steps: 是否提取推理步骤
        **kwargs: 其他参数
        
    返回:
        包含推理步骤的响应字典
        
    示例:
        # 数学推理任务
        response = reasoning_completion(
            messages=[{"role": "user", "content": "解这个方程：3x + 7 = 22"}],
            provider="phi4-transformers",
            extract_reasoning_steps=True
        )
        
        # 逻辑推理任务
        response = reasoning_completion(
            messages=[{"role": "user", "content": "如果所有的猫都是动物，而小花是一只猫，那么小花是什么？"}],
            provider="phi4-transformers",
            temperature=0.2
        )
    """
    # 准备参数
    params = kwargs.copy()
    if model is not None:
        params["model_name"] = model
    if device is not None:
        params["device"] = device
    if use_4bit_quantization:
        params["use_4bit_quantization"] = use_4bit_quantization
    if device_map != "auto":
        params["device_map"] = device_map
    if not trust_remote_code:
        params["trust_remote_code"] = trust_remote_code
    
    return _router.reasoning_completion(
        messages=messages,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        model=model,
        extract_reasoning_steps=extract_reasoning_steps,
        **params
    )


def get_model_info(provider: str = "mlx", **kwargs) -> Dict[str, Any]:
    """
    获取模型信息
    
    参数:
        provider: provider名称
        **kwargs: 构造函数参数
        
    返回:
        模型信息字典
    """
    return _router.get_model_info(provider, **kwargs)


def get_available_providers() -> Dict[str, str]:
    """
    获取所有可用的LLM提供者
    
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


def clear_cache():
    """清理缓存的LLM实例"""
    _router.clear_cache()

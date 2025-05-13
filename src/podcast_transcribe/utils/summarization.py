"""
文本摘要工具模块
支持多种摘要模型，包括Hugging Face Transformers、本地Qwen模型和MLX Qwen模型
"""

import os
import logging
from typing import Dict, Any, Optional, Union, List, Literal
from abc import ABC, abstractmethod

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# 配置日志
logger = logging.getLogger(__name__)

# 摘要器类型
SummarizerType = Literal["transformers", "qwen", "mlx_qwen"]


class AbstractSummarizer(ABC):
    """摘要器抽象基类"""
    
    @abstractmethod
    def summarize(
        self, 
        text: str, 
        max_length: int = 150, 
        min_length: int = 30,
        **kwargs
    ) -> str:
        """
        对文本进行摘要
        
        参数:
            text: 要摘要的文本
            max_length: 最大摘要长度
            min_length: 最小摘要长度
            **kwargs: 额外参数
            
        返回:
            摘要文本
        """
        pass


class TransformersSummarizer(AbstractSummarizer):
    """使用Hugging Face Transformers进行文本摘要的类"""

    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        初始化Transformers摘要器
        
        参数:
            model_name: 摘要模型名称
        """
        self.model_name = model_name
        self._summarizer = None
    
    @property
    def summarizer(self):
        """懒加载摘要模型"""
        if self._summarizer is None:
            try:
                logger.info(f"加载Transformers摘要模型: {self.model_name}")
                self._summarizer = pipeline("summarization", model=self.model_name)
            except Exception as e:
                logger.error(f"无法加载摘要模型: {str(e)}")
                raise RuntimeError(f"无法加载摘要模型: {str(e)}")
        return self._summarizer
    
    def summarize(
        self, 
        text: str, 
        max_length: int = 150, 
        min_length: int = 30,
        do_sample: bool = False,
        **kwargs
    ) -> str:
        """
        对文本进行摘要
        
        参数:
            text: 要摘要的文本
            max_length: 最大摘要长度
            min_length: 最小摘要长度
            do_sample: 是否使用采样
            **kwargs: 额外参数
            
        返回:
            摘要文本
        """
        if not text:
            return ""
            
        try:
            # 如果文本太长，需要分块处理
            if len(text) > 1024:
                return self._summarize_long_text(
                    text, 
                    max_length=max_length, 
                    min_length=min_length, 
                    do_sample=do_sample,
                    **kwargs
                )
                
            summary = self.summarizer(
                text, 
                max_length=max_length, 
                min_length=min_length, 
                do_sample=do_sample,
                **kwargs
            )
            return summary[0]['summary_text']
        except Exception as e:
            logger.error(f"摘要生成失败: {str(e)}")
            raise RuntimeError(f"摘要生成失败: {str(e)}")
    
    def _summarize_long_text(
        self, 
        text: str, 
        max_length: int = 150, 
        min_length: int = 30, 
        do_sample: bool = False,
        chunk_size: int = 1000,
        overlap: int = 100,
        **kwargs
    ) -> str:
        """
        对长文本进行分块摘要
        
        参数:
            text: 要摘要的文本
            max_length: 最大摘要长度
            min_length: 最小摘要长度
            do_sample: 是否使用采样
            chunk_size: 每个块的大小（字符数）
            overlap: 块之间的重叠（字符数）
            **kwargs: 额外参数
            
        返回:
            摘要文本
        """
        # 将文本分成多个块
        words = text.split()
        chunks = []
        chunk = []
        chunk_word_count = 0
        
        for word in words:
            chunk.append(word)
            chunk_word_count += 1
            
            if chunk_word_count >= chunk_size:
                chunks.append(' '.join(chunk))
                # 保留一部分重叠
                overlap_words = min(overlap, len(chunk))
                chunk = chunk[-overlap_words:]
                chunk_word_count = overlap_words
        
        # 添加最后一个块（如果有的话）
        if chunk:
            chunks.append(' '.join(chunk))
        
        # 摘要每个块
        summaries = []
        for chunk_text in chunks:
            chunk_summary = self.summarizer(
                chunk_text, 
                max_length=max(30, max_length // len(chunks)), 
                min_length=min(15, min_length // len(chunks)), 
                do_sample=do_sample,
                **kwargs
            )
            summaries.append(chunk_summary[0]['summary_text'])
        
        # 合并所有摘要
        combined_summary = ' '.join(summaries)
        
        # 如果合并后的摘要仍然太长，再次摘要
        if len(combined_summary) > max_length * 2:
            return self.summarize(
                combined_summary, 
                max_length=max_length, 
                min_length=min_length, 
                do_sample=do_sample,
                **kwargs
            )
            
        return combined_summary


class QwenSummarizer(AbstractSummarizer):
    """使用Qwen本地模型进行文本摘要的类"""

    def __init__(
        self, 
        model_path: str = "Qwen/Qwen1.5-1.8B",
        device: str = "cuda",
        load_in_4bit: bool = True,
        prompt_template: str = None
    ):
        """
        初始化Qwen摘要器
        
        参数:
            model_path: 模型路径，可以是HuggingFace模型ID或本地路径
            device: 推理设备，"cuda"或"cpu"
            load_in_4bit: 是否使用4bit量化加载模型以节省内存
            prompt_template: 自定义提示模板，使用{text}、{min_length}和{max_length}作为占位符
        """
        self.model_path = model_path
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.prompt_template = prompt_template or """请你为以下文本生成一个简洁、信息丰富的摘要，最少{min_length}字，最多{max_length}字。

原文：
{text}

摘要："""
        self._tokenizer = None
        self._model = None

        # 打印使用的device
        print(f"使用device: {self.device}，load_in_4bit: {self.load_in_4bit}")
    
    def _load_model(self):
        """加载模型和分词器"""
        if self._model is None or self._tokenizer is None:
            try:
                logger.info(f"加载Qwen模型: {self.model_path}")
                
                # 量化配置
                quantization_config = None
                if self.load_in_4bit:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype="float16"
                    )
                
                # 加载分词器
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, 
                    trust_remote_code=True
                )
                
                # 加载模型
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                    quantization_config=quantization_config
                )
                
                # 如果是CPU推理并且没有使用量化
                if self.device == "cpu" and not self.load_in_4bit:
                    self._model = self._model.to("cpu")
                
            except Exception as e:
                logger.error(f"无法加载Qwen模型: {str(e)}")
                raise RuntimeError(f"无法加载Qwen模型: {str(e)}")
    
    def summarize(
        self, 
        text: str, 
        max_length: int = 150, 
        min_length: int = 30,
        temperature: float = 0.7,
        top_p: float = 0.9,
        prompt_template: str = None,
        **kwargs
    ) -> str:
        """
        使用Qwen模型对文本进行摘要
        
        参数:
            text: 要摘要的文本
            max_length: 最大摘要长度
            min_length: 最小摘要长度
            temperature: 生成温度，控制随机性
            top_p: 核采样的概率阈值
            prompt_template: 临时覆盖默认提示模板
            **kwargs: 额外参数
            
        返回:
            摘要文本
        """
        if not text:
            return ""
        
        try:
            # 懒加载模型
            self._load_model()
            
            # 如果文本太长，分块处理
            if len(text) > 4000:
                return self._summarize_long_text(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    temperature=temperature,
                    top_p=top_p,
                    prompt_template=prompt_template,
                    **kwargs
                )
            
            # 使用提供的模板或默认模板
            template = prompt_template or self.prompt_template
            
            # 构建提示
            prompt = template.format(
                text=text,
                min_length=min_length,
                max_length=max_length
            )
            
            # 进行推理
            inputs = self._tokenizer(prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            # 生成摘要
            generation_kwargs = {
                "max_new_tokens": max_length * 2,  # 字符数到token数的估算
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": temperature > 0,
                "pad_token_id": self._tokenizer.eos_token_id,
                **kwargs
            }
            
            with torch.no_grad():
                outputs = self._model.generate(**inputs, **generation_kwargs)
            
            # 解码并提取摘要部分
            full_output = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取摘要（假设格式为"摘要："之后的文本）
            if "摘要：" in full_output:
                summary = full_output.split("摘要：")[-1].strip()
            else:
                # 如果没有找到"摘要："标记，返回生成的全部内容与提示后的部分
                prompt_length = len(prompt)
                if len(full_output) > prompt_length:
                    summary = full_output[prompt_length:].strip()
                else:
                    summary = full_output.strip()
            
            return summary
            
        except Exception as e:
            logger.error(f"Qwen摘要生成失败: {str(e)}")
            raise RuntimeError(f"Qwen摘要生成失败: {str(e)}")
    
    def _summarize_long_text(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
        temperature: float = 0.7,
        top_p: float = 0.9,
        chunk_size: int = 3000,
        overlap: int = 300,
        prompt_template: str = None,
        **kwargs
    ) -> str:
        """
        对长文本进行分块摘要
        
        参数:
            text: 要摘要的文本
            max_length: 最大摘要长度
            min_length: 最小摘要长度
            temperature: 生成温度
            top_p: 核采样的概率阈值
            chunk_size: 每个块的大小（字符数）
            overlap: 块之间的重叠（字符数）
            prompt_template: 临时覆盖默认提示模板
            **kwargs: 额外参数
            
        返回:
            摘要文本
        """
        # 将文本分成多个块
        chars = list(text)
        chunks = []
        current_chunk = []
        
        for char in chars:
            current_chunk.append(char)
            
            if len(current_chunk) >= chunk_size:
                chunks.append(''.join(current_chunk))
                # 保留一部分重叠
                overlap_chars = min(overlap, len(current_chunk))
                current_chunk = current_chunk[-overlap_chars:]
        
        # 添加最后一个块（如果有的话）
        if current_chunk:
            chunks.append(''.join(current_chunk))
        
        # 使用提供的模板或默认模板
        template = prompt_template or self.prompt_template
        
        # 摘要每个块
        summaries = []
        for chunk_text in chunks:
            # 构建块的提示
            chunk_prompt = template.format(
                text=chunk_text,
                min_length=min(15, min_length // len(chunks)),
                max_length=max(30, max_length // len(chunks))
            )
            
            # 进行推理
            inputs = self._tokenizer(chunk_prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            # 生成摘要
            generation_kwargs = {
                "max_new_tokens": (max_length // len(chunks)) * 2,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": temperature > 0,
                "pad_token_id": self._tokenizer.eos_token_id,
                **kwargs
            }
            
            with torch.no_grad():
                outputs = self._model.generate(**inputs, **generation_kwargs)
            
            # 解码并提取摘要部分
            full_output = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取摘要（假设格式为"摘要："之后的文本）
            if "摘要：" in full_output:
                chunk_summary = full_output.split("摘要：")[-1].strip()
            else:
                # 如果没有找到"摘要："标记，返回生成的全部内容与提示后的部分
                prompt_length = len(chunk_prompt)
                if len(full_output) > prompt_length:
                    chunk_summary = full_output[prompt_length:].strip()
                else:
                    chunk_summary = full_output.strip()
                    
            summaries.append(chunk_summary)
        
        # 合并所有摘要
        combined_summary = ''.join(summaries)
        
        # 如果合并后的摘要仍然太长，再次摘要
        if len(combined_summary) > max_length * 2:
            return self.summarize(
                combined_summary,
                max_length=max_length,
                min_length=min_length,
                temperature=temperature,
                top_p=top_p,
                prompt_template=prompt_template,
                **kwargs
            )
            
        return combined_summary


class MLXQwenSummarizer(AbstractSummarizer):
    """使用MLX框架的Qwen模型进行文本摘要的类，专为Apple Silicon芯片优化"""

    def __init__(
        self, 
        model_path: str = "mlx-community/Qwen3-4B-4bit",
        prompt_template: str = None
    ):
        """
        初始化MLX Qwen摘要器
        
        参数:
            model_path: 模型路径，可以是HuggingFace模型ID或本地路径
            prompt_template: 自定义提示模板，使用{text}、{min_length}和{max_length}作为占位符
        """
        self.model_path = model_path
        self.prompt_template = prompt_template or """请你为以下文本生成一个简洁、信息丰富的摘要，最少{min_length}字，最多{max_length}字。

原文：
{text}

摘要："""
        self._tokenizer = None
        self._model = None
        self._processor = None

        # 打印使用MLX
        print(f"使用MLX框架和模型: {self.model_path}")
    
    def _load_model(self):
        """加载MLX模型和分词器"""
        if self._model is None or self._tokenizer is None:
            try:
                logger.info(f"加载MLX Qwen模型: {self.model_path}")
                
                # 导入MLX相关库
                try:
                    import mlx.core as mx
                    from mlx_lm import load, generate
                except ImportError:
                    logger.error("MLX或mlx_lm库未安装，请先安装 'pip install mlx mlx_lm'")
                    raise ImportError("请先安装MLX: 'pip install mlx mlx_lm'")
                
                # 加载MLX模型 - 修复：load函数返回的是(model, tokenizer)元组
                self._model, self._tokenizer = load(self.model_path)
                self._processor = None  # 不再需要processor
                
            except Exception as e:
                logger.error(f"无法加载MLX Qwen模型: {str(e)}")
                raise RuntimeError(f"无法加载MLX Qwen模型: {str(e)}")
    
    def summarize(
        self, 
        text: str, 
        max_length: int = 150, 
        min_length: int = 30,
        temperature: float = 0.7,
        top_p: float = 0.9,
        prompt_template: str = None,
        **kwargs
    ) -> str:
        """
        使用MLX Qwen模型对文本进行摘要
        
        参数:
            text: 要摘要的文本
            max_length: 最大摘要长度
            min_length: 最小摘要长度
            temperature: 生成温度，控制随机性
            top_p: 核采样的概率阈值
            prompt_template: 临时覆盖默认提示模板
            **kwargs: 额外参数
            
        返回:
            摘要文本
        """
        if not text:
            return ""
        
        try:
            # 懒加载模型
            self._load_model()
            
            # 如果文本太长，分块处理
            if len(text) > 4000:
                return self._summarize_long_text(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    temperature=temperature,
                    top_p=top_p,
                    prompt_template=prompt_template,
                    **kwargs
                )
            
            # 使用提供的模板或默认模板
            template = prompt_template or self.prompt_template
            
            # 构建提示
            prompt = template.format(
                text=text,
                min_length=min_length,
                max_length=max_length
            )
            
            # 导入MLX生成模块
            import mlx.core as mx
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler
            
            # 创建采样器
            sampler = make_sampler(temp=temperature, top_p=top_p)
            
            # 生成摘要
            generation_kwargs = {
                "max_tokens": max_length * 2,  # 字符数到token数的估算
                "sampler": sampler,  # 使用采样器而不是直接传递temperature
                **kwargs
            }
            
            # 使用MLX进行生成 - 直接使用model和tokenizer，不再使用processor
            outputs = generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                **generation_kwargs
            )
            
            # 提取摘要（假设格式为"摘要："之后的文本）
            if "摘要：" in outputs:
                summary = outputs.split("摘要：")[-1].strip()
            else:
                # 如果没有找到"摘要："标记，返回生成的全部内容与提示后的部分
                prompt_length = len(prompt)
                if len(outputs) > prompt_length:
                    summary = outputs[prompt_length:].strip()
                else:
                    summary = outputs.strip()
            
            return summary
            
        except Exception as e:
            logger.error(f"MLX Qwen摘要生成失败: {str(e)}")
            raise RuntimeError(f"MLX Qwen摘要生成失败: {str(e)}")
    
    def _summarize_long_text(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
        temperature: float = 0.7,
        top_p: float = 0.9,
        chunk_size: int = 3000,
        overlap: int = 300,
        prompt_template: str = None,
        **kwargs
    ) -> str:
        """
        对长文本进行分块摘要
        
        参数:
            text: 要摘要的文本
            max_length: 最大摘要长度
            min_length: 最小摘要长度
            temperature: 生成温度
            top_p: 核采样的概率阈值
            chunk_size: 每个块的大小（字符数）
            overlap: 块之间的重叠（字符数）
            prompt_template: 临时覆盖默认提示模板
            **kwargs: 额外参数
            
        返回:
            摘要文本
        """
        # 将文本分成多个块
        chars = list(text)
        chunks = []
        current_chunk = []
        
        for char in chars:
            current_chunk.append(char)
            
            if len(current_chunk) >= chunk_size:
                chunks.append(''.join(current_chunk))
                # 保留一部分重叠
                overlap_chars = min(overlap, len(current_chunk))
                current_chunk = current_chunk[-overlap_chars:]
        
        # 添加最后一个块（如果有的话）
        if current_chunk:
            chunks.append(''.join(current_chunk))
        
        # 使用提供的模板或默认模板
        template = prompt_template or self.prompt_template
        
        # 摘要每个块
        summaries = []
        for chunk_text in chunks:
            # 构建块的提示
            chunk_prompt = template.format(
                text=chunk_text,
                min_length=min(15, min_length // len(chunks)),
                max_length=max(30, max_length // len(chunks))
            )
            
            # 导入MLX生成模块
            import mlx.core as mx
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler
            
            # 创建采样器
            sampler = make_sampler(temp=temperature, top_p=top_p)
            
            # 生成摘要
            generation_kwargs = {
                "max_tokens": (max_length // len(chunks)) * 2,
                "sampler": sampler,  # 使用采样器而不是直接传递temperature
                **kwargs
            }
            
            # 使用MLX进行生成 - 直接使用model和tokenizer，不再使用processor
            outputs = generate(
                self._model,
                self._tokenizer,
                prompt=chunk_prompt,
                **generation_kwargs
            )
            
            # 提取摘要（假设格式为"摘要："之后的文本）
            if "摘要：" in outputs:
                chunk_summary = outputs.split("摘要：")[-1].strip()
            else:
                # 如果没有找到"摘要："标记，返回生成的全部内容与提示后的部分
                prompt_length = len(chunk_prompt)
                if len(outputs) > prompt_length:
                    chunk_summary = outputs[prompt_length:].strip()
                else:
                    chunk_summary = outputs.strip()
                    
            summaries.append(chunk_summary)
        
        # 合并所有摘要
        combined_summary = ''.join(summaries)
        
        # 如果合并后的摘要仍然太长，再次摘要
        if len(combined_summary) > max_length * 2:
            return self.summarize(
                combined_summary,
                max_length=max_length,
                min_length=min_length,
                temperature=temperature,
                top_p=top_p,
                prompt_template=prompt_template,
                **kwargs
            )
            
        return combined_summary


class TextSummarizer:
    """文本摘要器工厂类，支持多种摘要模型"""
    
    def __init__(
        self, 
        summarizer_type: SummarizerType = "transformers",
        model_name_or_path: str = "facebook/bart-large-cnn",
        device: str = "cuda",
        load_in_4bit: bool = True,
        prompt_template: str = None,
        **kwargs
    ):
        """
        初始化文本摘要器
        
        参数:
            summarizer_type: 摘要器类型，支持"transformers"、"qwen"和"mlx_qwen"
            model_name_or_path: 模型名称或路径
            device: 推理设备，"cuda"、"cpu"或"mps"（仅对Qwen有效，MLX自动使用Apple Silicon）
            load_in_4bit: 是否使用4bit量化加载模型（仅对Qwen有效）
            prompt_template: 自定义提示模板（对Qwen和MLX有效）
            **kwargs: 其他参数
        """
        self.summarizer_type = summarizer_type
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.prompt_template = prompt_template
        self.kwargs = kwargs
        self._summarizer = None
        
    @property
    def summarizer(self) -> AbstractSummarizer:
        """懒加载摘要模型"""
        if self._summarizer is None:
            if self.summarizer_type == "transformers":
                self._summarizer = TransformersSummarizer(model_name=self.model_name_or_path)
            elif self.summarizer_type == "qwen":
                self._summarizer = QwenSummarizer(
                    model_path=self.model_name_or_path,
                    device=self.device,
                    load_in_4bit=self.load_in_4bit,
                    prompt_template=self.prompt_template
                )
            elif self.summarizer_type == "mlx_qwen":
                self._summarizer = MLXQwenSummarizer(
                    model_path=self.model_name_or_path,
                    prompt_template=self.prompt_template
                )
            else:
                raise ValueError(f"不支持的摘要器类型: {self.summarizer_type}")
        return self._summarizer
    
    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
        **kwargs
    ) -> str:
        """
        对文本进行摘要
        
        参数:
            text: 要摘要的文本
            max_length: 最大摘要长度
            min_length: 最小摘要长度
            **kwargs: 额外参数，会覆盖初始化时的参数
            
        返回:
            摘要文本
        """
        # 合并参数
        all_kwargs = {**self.kwargs, **kwargs}
        
        return self.summarizer.summarize(
            text=text,
            max_length=max_length,
            min_length=min_length,
            **all_kwargs
        ) 
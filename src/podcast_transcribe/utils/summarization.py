"""
文本摘要工具模块
"""

from typing import Dict, Any, Optional
from transformers import pipeline


class TextSummarizer:
    """使用Hugging Face Transformers进行文本摘要的类"""

    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        初始化文本摘要器
        
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
                self._summarizer = pipeline("summarization", model=self.model_name)
            except Exception as e:
                raise RuntimeError(f"无法加载摘要模型: {str(e)}")
        return self._summarizer
    
    def summarize(
        self, 
        text: str, 
        max_length: int = 150, 
        min_length: int = 30, 
        do_sample: bool = False
    ) -> str:
        """
        对文本进行摘要
        
        参数:
            text: 要摘要的文本
            max_length: 最大摘要长度
            min_length: 最小摘要长度
            do_sample: 是否使用采样
            
        返回:
            摘要文本
        """
        if not text:
            return ""
            
        try:
            # 如果文本太长，需要分块处理
            if len(text) > 1024:
                return self._summarize_long_text(text, max_length, min_length, do_sample)
                
            summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=do_sample)
            return summary[0]['summary_text']
        except Exception as e:
            raise RuntimeError(f"摘要生成失败: {str(e)}")
    
    def _summarize_long_text(
        self, 
        text: str, 
        max_length: int = 150, 
        min_length: int = 30, 
        do_sample: bool = False,
        chunk_size: int = 1000,
        overlap: int = 100
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
                do_sample=do_sample
            )
            summaries.append(chunk_summary[0]['summary_text'])
        
        # 合并所有摘要
        combined_summary = ' '.join(summaries)
        
        # 如果合并后的摘要仍然太长，再次摘要
        if len(combined_summary) > max_length * 2:
            return self.summarize(combined_summary, max_length, min_length, do_sample)
            
        return combined_summary 
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用MLX框架的Qwen3模型为转录文件生成摘要，专为Apple Silicon优化
"""

import os
import logging
from pathlib import Path
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.podcast_transcribe.utils.summarization import TextSummarizer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mlx_qwen_text_summarize_example")

def main():
    """主函数"""
    transcript_file = Path.joinpath(Path(__file__).parent.parent, "output", "lex_ai_john_carmack_1_transcript.txt")
    output_file = Path.joinpath(Path(__file__).parent.parent, "output", "lex_ai_john_carmack_1_summary-mlx-qwen.txt")
    mlx_model_path = "mlx-community/Qwen3-8B-4bit"  # 使用HuggingFace的MLX优化版本

    # 自定义提示模板
    prompt_template = "Please summarize the following podcast conversation as briefly as possible. In Chinese.：\n\n{text}"

    # 验证输入文件存在
    if not os.path.isfile(transcript_file):
        logger.error(f"转录文件不存在: {transcript_file}")
        return
    
    # 读取转录文件
    logger.info(f"读取转录文件: {transcript_file}")
    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript_text = f.read()
    except Exception as e:
        logger.error(f"无法读取转录文件: {str(e)}")
        return
    
    if not transcript_text.strip():
        logger.error("转录文件内容为空")
        return
    
    # 初始化MLX Qwen摘要器
    logger.info(f"初始化MLX Qwen摘要器，使用模型: {mlx_model_path}")
    summarizer = TextSummarizer(
        summarizer_type="mlx_qwen",
        model_name_or_path=mlx_model_path,
        prompt_template=prompt_template
    )
    
    # 生成摘要参数
    summary_kwargs = {
        'temperature': 0.7,
        'top_p': 0.9,
        'max_length': 300,
        'min_length': 100,
    }
    
    # 生成摘要
    logger.info("开始使用MLX生成摘要...")
    try:
        summary = summarizer.summarize(
            text=transcript_text,
            **summary_kwargs
        )
    except Exception as e:
        logger.error(f"摘要生成失败: {str(e)}")
        return
    
    # 如果提供了输出文件，保存摘要
    if output_file:
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            logger.info(f"摘要已保存到: {output_file}")
        except Exception as e:
            logger.error(f"保存摘要到文件失败: {str(e)}")
    
    # 打印摘要
    print("\n===== MLX Qwen生成的摘要 =====")
    print(summary)
    print("==============================\n")


if __name__ == "__main__":
    main() 
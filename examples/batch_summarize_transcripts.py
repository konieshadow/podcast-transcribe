#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量处理output目录下的所有转录文件并使用Qwen模型生成摘要
"""

import os
import argparse
import logging
import glob
from pathlib import Path

from src.podcast_transcribe.utils.summarization import TextSummarizer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("batch_summarize")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='批量处理转录文件并生成摘要'
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        default='output',
        help='转录文件所在目录（默认为output）'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*transcript*.txt',
        help='转录文件匹配模式（默认为*transcript*.txt）'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output/summaries',
        help='摘要输出目录（默认为output/summaries）'
    )
    
    parser.add_argument(
        '--qwen_model_path',
        type=str,
        default="Qwen/Qwen1.5-1.8B",
        help='Qwen模型路径（默认为Qwen/Qwen1.5-1.8B）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda',
        help='推理设备（默认为cuda）'
    )
    
    parser.add_argument(
        '--no_4bit_quantization',
        action='store_true',
        help='禁用4bit量化加载模型'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='生成温度，控制随机性（默认为0.7）'
    )
    
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='核采样的概率阈值（默认为0.9）'
    )
    
    parser.add_argument(
        '--max_length',
        type=int,
        default=150,
        help='最大摘要长度（默认150）'
    )
    
    parser.add_argument(
        '--min_length',
        type=int,
        default=30,
        help='最小摘要长度（默认30）'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 确保输入目录存在
    if not os.path.isdir(args.input_dir):
        logger.error(f"输入目录不存在: {args.input_dir}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 查找所有匹配的转录文件
    pattern = os.path.join(args.input_dir, args.pattern)
    transcript_files = glob.glob(pattern)
    
    if not transcript_files:
        logger.error(f"未找到匹配的转录文件，匹配模式: {pattern}")
        return
    
    logger.info(f"找到 {len(transcript_files)} 个转录文件")
    
    # 初始化Qwen摘要器
    logger.info(f"初始化Qwen摘要器，使用模型: {args.qwen_model_path}")
    summarizer = TextSummarizer(
        summarizer_type="qwen",
        model_name_or_path=args.qwen_model_path,
        device=args.device,
        load_in_4bit=not args.no_4bit_quantization
    )
    
    # 生成摘要参数
    summary_kwargs = {
        'temperature': args.temperature,
        'top_p': args.top_p
    }
    
    # 处理每个转录文件
    for i, transcript_file in enumerate(transcript_files, 1):
        file_basename = os.path.basename(transcript_file)
        file_name = os.path.splitext(file_basename)[0]
        
        # 替换文件名中的transcript为summary
        summary_basename = file_name.replace('transcript', 'summary') + '.txt'
        if summary_basename == file_basename:  # 如果没有替换成功
            summary_basename = f"{file_name}_summary.txt"
        
        summary_file = os.path.join(args.output_dir, summary_basename)
        
        logger.info(f"处理文件 [{i}/{len(transcript_files)}]: {file_basename}")
        
        # 读取转录文件
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
        except Exception as e:
            logger.error(f"无法读取转录文件: {str(e)}")
            continue
        
        if not transcript_text.strip():
            logger.warning(f"转录文件内容为空: {file_basename}")
            continue
        
        # 生成摘要
        try:
            summary = summarizer.summarize(
                text=transcript_text,
                max_length=args.max_length,
                min_length=args.min_length,
                **summary_kwargs
            )
        except Exception as e:
            logger.error(f"摘要生成失败: {str(e)}")
            continue
        
        # 保存摘要
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            logger.info(f"摘要已保存到: {summary_file}")
        except Exception as e:
            logger.error(f"保存摘要到文件失败: {str(e)}")
    
    logger.info("批处理完成")


if __name__ == "__main__":
    main() 
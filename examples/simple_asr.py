#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的语音识别示例程序
从本地文件读取音频并进行转录
"""

import logging
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.podcast_transcribe.audio import load_audio
from src.podcast_transcribe.asr.asr_router import transcribe_audio

logger = logging.getLogger("asr_example")


def main():
    """主函数"""
    audio_file = Path.joinpath(Path(__file__).parent, "input", "lex_ai_john_carmack_1.wav")  # 播客音频文件路径
    # audio_file = "/Users/konie/Desktop/voices/lex_ai_john_carmack_30.wav"  # 播客音频文件路径
    # audio_file = "/Users/konie/Desktop/voices/podcast1_1.wav"
    # model = "distil-whisper"
    model = "distil-whisper/distil-large-v3.5"

    device = "mps"
    
    # 检查文件是否存在
    if not os.path.exists(audio_file):
        print(f"错误：文件 '{audio_file}' 不存在")
        return 1
    
    try:
        print(f"正在加载音频文件: {audio_file}")
        # 加载音频文件
        audio, _ = load_audio(audio_file)
        
        print(f"音频信息: 时长={audio.duration_seconds:.2f}秒, 通道数={audio.channels}, 采样率={audio.frame_rate}Hz")
        
        # 进行转录
        print("开始转录...")
        result = transcribe_audio(audio, provider="distil_whisper_transformers", model_name=model, device=device)
        
        # 输出结果
        print("\n转录结果:")
        print("-" * 50)
        print(f"检测到的语言: {result.language}")
        print(f"完整文本: {result.text}")
        print("\n分段信息:")
        for i, segment in enumerate(result.segments, 1):
            print(f"分段 {i}: [{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
        
        return 0
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return 1


if __name__ == "__main__":
    main()

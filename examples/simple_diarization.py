#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的说话人标注示例程序
从本地文件读取音频并进行说话人分离
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.podcast_transcribe.audio import load_audio
from src.podcast_transcribe.diarization.diarization_pyannote import diarize_audio


def main():
    """主函数"""
    audio_file = Path.joinpath(Path(__file__).parent, "input", "lex_ai_john_carmack_1.wav")  # 播客音频文件路径
    model_name = "pyannote/speaker-diarization-3.1"  # 说话人分离模型名称
    hf_token = "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"  # Hugging Face API 令牌
    device = "mps"  # 设备类型
    
    # 检查文件是否存在
    if not os.path.exists(audio_file):
        print(f"错误：文件 '{audio_file}' 不存在")
        return 1
    
    # 检查令牌是否设置
    if not hf_token:
        print("错误：未设置HF_TOKEN环境变量，请设置后再运行")
        return 1
    
    try:
        print(f"正在加载音频文件: {audio_file}")
        # 加载音频文件
        audio, _ = load_audio(audio_file)
        
        print(f"音频信息: 时长={audio.duration_seconds:.2f}秒, 通道数={audio.channels}, 采样率={audio.frame_rate}Hz")
        
        # 进行说话人分离
        print("开始说话人分离...")
        result = diarize_audio(audio, model_name=model_name, token=hf_token, device=device)
        
        # 输出结果
        print("\n说话人分离结果:")
        print("-" * 50)
        print(f"检测到的说话人数量: {result.num_speakers}")
        print(f"分段总数: {len(result.segments)}")
        
        print("\n分段详情:")
        for i, segment in enumerate(result.segments, 1):
            start = segment["start"]
            end = segment["end"]
            speaker = segment["speaker"]
            duration = end - start
            print(f"分段 {i}: [{start:.2f}s - {end:.2f}s] (时长: {duration:.2f}s) 说话人: {speaker}")
        
        return 0
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

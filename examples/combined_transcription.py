#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
整合ASR和说话人分离的示例程序
从本地文件读取音频，同时进行转录和说话人分离
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入必要的模块，使用正确的导入路径
from src.podcast_transcribe.audio import load_audio
from src.podcast_transcribe.transcriber import transcribe_audio


def main():
    """主函数"""
    audio_file = Path.joinpath(Path(__file__).parent, "input", "lex_ai_john_carmack_1.wav")  # 播客音频文件路径
    # audio_file = "/Users/konie/Desktop/voices/lex_ai_john_carmack_30.wav"
    
    # 模型配置
    asr_model_name = "mlx-community/parakeet-tdt-0.6b-v2"  # ASR模型名称
    diarization_model_name = "pyannote/speaker-diarization-3.1"  # 说话人分离模型名称
    hf_token = "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"  # Hugging Face API 令牌
    device = "mps"  # 设备类型
    
    # 处理参数
    chunk_size_ms = 30000  # 30秒块
    overlap_ms = 5000  # 5秒重叠
    
    # 检查文件是否存在
    if not os.path.exists(audio_file):
        print(f"错误：文件 '{audio_file}' 不存在")
        return 1
    
    # 检查HF令牌
    if not hf_token:
        print("警告：未设置HF_TOKEN环境变量，必须设置此环境变量才能使用pyannote说话人分离模型")
        print("请执行：export HF_TOKEN='你的HuggingFace令牌'")
        return 1
    
    try:
        print(f"正在加载音频文件: {audio_file}")
        # 加载音频文件
        audio, _ = load_audio(audio_file)
        
        print(f"音频信息: 时长={audio.duration_seconds:.2f}秒, 通道数={audio.channels}, 采样率={audio.frame_rate}Hz")
      
        result = transcribe_audio(
            audio, 
            asr_model_name=asr_model_name,
            diarization_model_name=diarization_model_name,
            hf_token=hf_token,
            device=device,
        )
        
        # 输出结果
        print("\n转录结果:")
        print("-" * 50)
        print(f"检测到的语言: {result.language}")
        print(f"检测到的说话人数量: {result.num_speakers}")
        print(f"总文本长度: {len(result.text)} 字符")
        
        # 输出每个说话人的部分
        speakers = set(segment.speaker for segment in result.segments)
        for speaker in sorted(speakers):
            speaker_segments = [seg for seg in result.segments if seg.speaker == speaker]
            total_duration = sum(seg.end - seg.start for seg in speaker_segments)
            print(f"\n说话人 {speaker}: 共 {len(speaker_segments)} 个片段, 总时长 {total_duration:.2f} 秒")
        
        # 输出详细分段信息
        print("\n详细分段信息:")
        for i, segment in enumerate(result.segments, 1):
            if i <= 20 or i > len(result.segments) - 20:  # 仅显示前20个和后20个分段
                print(f"段落 {i}/{len(result.segments)}: [{segment.start:.2f}s - {segment.end:.2f}s] 说话人: {segment.speaker} 文本: {segment.text}")
            elif i == 21:
                print("... 省略中间部分 ...")
        
        return 0
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
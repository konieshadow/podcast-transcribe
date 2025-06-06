#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
整合ASR和说话人分离的示例程序
从本地文件读取音频，同时进行转录和说话人分离
"""

import json
import os
import sys
from pathlib import Path
from dataclasses import asdict

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入必要的模块，使用正确的导入路径
from src.podcast_transcribe.audio import load_audio
from src.podcast_transcribe.transcriber import transcribe_audio


def main():
    """主函数"""
    audio_file = Path.joinpath(Path(__file__).parent, "input", "lex_ai_john_carmack_1.wav")  # 播客音频文件路径
    # audio_file = Path("/Users/konie/Desktop/voices/lex_ai_john_carmack_30.wav")
    
    # 模型配置
    asr_model_name = "distil-whisper/distil-large-v3.5"  # ASR模型名称
    diarization_model_name = "pyannote/speaker-diarization-3.1"  # 说话人分离模型名称
    device = "mps"  # 设备类型
    segmentation_batch_size = 64
    parallel = True

    # 检查文件是否存在
    if not os.path.exists(audio_file):
        print(f"错误：文件 '{audio_file}' 不存在")
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
            device=device,
            segmentation_batch_size=segmentation_batch_size,
            parallel=parallel,
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

        # 将转录结果保存为json文件，文件名取自音频文件名
        output_file = Path.joinpath(Path(__file__).parent, "output", f"{audio_file.stem}.transcription.json")
        # 创建上层文件夹
        output_dir = Path.joinpath(Path(__file__).parent, "output")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(asdict(result), f)
            print(f"转录结果已保存到 {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
"""
语音识别模块基类
"""

import os
import numpy as np
from pydub import AudioSegment
from typing import Dict, List, Union, Optional, Tuple
# from dataclasses import dataclass # dataclass is now imported from schemas if needed or already there
import logging

from ..schemas import TranscriptionResult # Added import

# 配置日志
logger = logging.getLogger("asr")


class BaseTranscriber:
    """统一的语音识别基类，支持MLX和Transformers等多种框架"""
    
    def __init__(
        self, 
        model_name: str,
        device: str = None,
    ):
        """
        初始化转录器
        
        参数:
            model_name: 模型名称
            device: 推理设备，'cpu'或'cuda'，对于MLX框架此参数可忽略
        """
        self.model_name = model_name
        self.device = device
        self.pipeline = None  # 用于Transformers
        self.model = None     # 用于MLX等其他框架
                
        logger.info(f"初始化转录器，模型: {model_name}" + (f"，设备: {device}" if device else ""))
        
        # 子类需要实现_load_model方法
        self._load_model()
        
    def _load_model(self):
        """
        加载模型（需要在子类中实现）
        """
        raise NotImplementedError("子类必须实现_load_model方法")
    
    def _prepare_audio(self, audio: AudioSegment) -> AudioSegment:
        """
        准备音频数据
        
        参数:
            audio: 输入的AudioSegment对象
            
        返回:
            处理后的AudioSegment对象
        """
        logger.debug(f"准备音频数据: 时长={len(audio)/1000:.2f}秒, 采样率={audio.frame_rate}Hz, 声道数={audio.channels}")
        
        # 确保采样率为16kHz
        if audio.frame_rate != 16000:
            logger.debug(f"重采样音频从 {audio.frame_rate}Hz 到 16000Hz")
            audio = audio.set_frame_rate(16000)
            
        # 确保是单声道
        if audio.channels > 1:
            logger.debug(f"将{audio.channels}声道音频转换为单声道")
            audio = audio.set_channels(1)
            
        logger.debug(f"音频处理完成")
        
        return audio
    
    def _detect_language(self, text: str) -> str:
        """
        简单的语言检测（基于经验规则）
        
        参数:
            text: 识别出的文本
            
        返回:
            检测到的语言代码
        """
        # 简单的规则检测，实际应用中应使用更准确的语言检测
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        chinese_ratio = chinese_chars / len(text) if text else 0
        logger.debug(f"语言检测: 中文字符比例 = {chinese_ratio:.2f}")
        
        if chinese_chars > len(text) * 0.3:
            return "zh"
        return "en"
    
    def _convert_segments(self, model_result) -> List[Dict[str, Union[float, str]]]:
        """
        将模型的分段结果转换为所需格式（需要在子类中实现）
        
        参数:
            model_result: 模型返回的结果
            
        返回:
            转换后的分段列表
        """
        raise NotImplementedError("子类必须实现_convert_segments方法")
    
    def transcribe(self, audio: AudioSegment, chunk_duration_s: int = 30, overlap_s: int = 5) -> TranscriptionResult:
        """
        转录音频，支持长音频分块处理。
        
        参数:
            audio: 要转录的AudioSegment对象
            chunk_duration_s: 分块处理的块时长（秒）。如果音频短于此，则不分块。
            overlap_s: 分块间的重叠时长（秒）。
            
        返回:
            TranscriptionResult对象，包含转录结果
        """
        logger.info(f"开始转录 {len(audio)/1000:.2f} 秒的音频。分块设置: 块时长={chunk_duration_s}s, 重叠={overlap_s}s")

        if overlap_s >= chunk_duration_s and len(audio)/1000.0 > chunk_duration_s :
            logger.error("重叠时长必须小于块时长。")
            raise ValueError("overlap_s 必须小于 chunk_duration_s。")

        total_duration_ms = len(audio)
        chunk_duration_ms = chunk_duration_s * 1000
        overlap_ms = overlap_s * 1000

        if total_duration_ms <= chunk_duration_ms:
            logger.debug("音频时长不大于设定块时长，直接进行完整转录。")
            processed_audio = self._prepare_audio(audio)
            samples = np.array(processed_audio.get_array_of_samples(), dtype=np.float32) / 32768.0
            
            try:
                model_result = self._perform_transcription(samples)
                text = self._get_text_from_result(model_result)
                segments = self._convert_segments(model_result)
                language = self._detect_language(text)
                
                logger.info(f"单块转录完成，语言: {language}，文本长度: {len(text)}，分段数: {len(segments)}")
                return TranscriptionResult(text=text, segments=segments, language=language)
            except Exception as e:
                logger.error(f"单块转录失败: {str(e)}", exc_info=True)
                raise RuntimeError(f"单块转录失败: {str(e)}")

        # 长音频分块处理
        final_segments = []
        # current_pos_ms 指的是当前块要处理的"新内容"的起始点在原始音频中的位置
        current_pos_ms = 0 

        while current_pos_ms < total_duration_ms:
            # 计算当前块实际送入模型处理的音频的起始和结束时间点
            # 对于第一个块，start_process_ms = 0
            # 对于后续块，start_process_ms 会向左回退 overlap_ms 以包含重叠区域
            start_process_ms = max(0, current_pos_ms - overlap_ms)
            end_process_ms = min(start_process_ms + chunk_duration_ms, total_duration_ms)

            # 如果计算出的块起始点已经等于或超过总时长，说明处理完毕
            if start_process_ms >= total_duration_ms:
                break
            
            chunk_audio = audio[start_process_ms:end_process_ms]
            
            logger.info(f"处理音频块: {start_process_ms/1000.0:.2f}s - {end_process_ms/1000.0:.2f}s (新内容起始于: {current_pos_ms/1000.0:.2f}s)")

            if len(chunk_audio) == 0:
                logger.warning(f"生成了一个空的音频块，跳过。起始: {start_process_ms/1000.0:.2f}s, 结束: {end_process_ms/1000.0:.2f}s")
                # 必须推进 current_pos_ms 以避免死循环
                advance_ms = chunk_duration_ms - overlap_ms
                if advance_ms <= 0: # 应该在函数开始时已检查 overlap_s < chunk_duration_s
                    raise RuntimeError("块推进时长配置错误，可能导致死循环。")
                current_pos_ms += advance_ms
                continue

            processed_chunk_audio = self._prepare_audio(chunk_audio)
            samples = np.array(processed_chunk_audio.get_array_of_samples(), dtype=np.float32) / 32768.0
            
            try:
                model_result = self._perform_transcription(samples)
                segments_chunk = self._convert_segments(model_result)

                for seg in segments_chunk:
                    # seg["start"] 和 seg["end"] 是相对于当前块 (chunk_audio) 的起始点（即0）
                    # 计算 segment 在原始完整音频中的绝对起止时间
                    global_seg_start_s = start_process_ms / 1000.0 + seg["start"]
                    global_seg_end_s = start_process_ms / 1000.0 + seg["end"]
                    
                    # 核心去重逻辑：
                    # 我们只接受那些真实开始于 current_pos_ms / 1000.0 之后的 segment。
                    # current_pos_ms 是当前块应该贡献的"新"内容的开始时间。
                    # 对于第一个块 (current_pos_ms == 0)，所有 segment 都被接受（只要它们的 start >= 0）。
                    # 对于后续块，只有当 segment 的全局开始时间 >= 当前块新内容的开始时间时，才添加。
                    if global_seg_start_s >= current_pos_ms / 1000.0:
                        final_segments.append({
                            "start": global_seg_start_s,
                            "end": global_seg_end_s,
                            "text": seg["text"]
                        })
                    # 特殊处理第一个块，因为 current_pos_ms 为 0，上面的条件 global_seg_start_s >= 0 总是满足。
                    # 但为了更清晰，如果不是第一个块，但 segment 跨越了 current_pos_ms,
                    # 它的起始部分在重叠区，结束部分在非重叠区。
                    # 当前逻辑是，如果它的 global_seg_start_s < current_pos_ms / 1000.0，它就被丢弃。
                    # 这是为了确保不重复记录重叠区域的开头部分。
                    # 如果一个 segment 完全在重叠区内且在前一个块已被记录，此逻辑可避免重复。

            except Exception as e:
                logger.error(f"处理音频块 {start_process_ms/1000.0:.2f}s - {end_process_ms/1000.0:.2f}s 失败: {str(e)}", exc_info=True)
            
            # 更新下一个"新内容"块的起始位置
            advance_ms = chunk_duration_ms - overlap_ms
            current_pos_ms += advance_ms
        
        # 对收集到的所有 segments 按开始时间排序
        final_segments.sort(key=lambda s: s["start"])

        # 可选：进一步清理 segments，例如合并非常接近且文本连续的，或移除完全重复的
        cleaned_segments = []
        if final_segments:
            cleaned_segments.append(final_segments[0])
            for i in range(1, len(final_segments)):
                prev_s = cleaned_segments[-1]
                curr_s = final_segments[i]
                # 简单的去重：如果时间戳和文本都几乎一样，则认为是重复
                if abs(curr_s["start"] - prev_s["start"]) < 0.01 and \
                   abs(curr_s["end"] - prev_s["end"]) < 0.01 and \
                   curr_s["text"] == prev_s["text"]:
                    continue
                
                # 如果当前 segment 的开始时间在前一个 segment 的结束时间之前，
                # 并且文本有明显重叠，可能需要更智能的合并。
                # 目前的逻辑通过 global_seg_start_s >= current_pos_ms / 1000.0 过滤，
                # 已经大大减少了直接的 segment 重复。
                # 此处的清理更多是处理模型在边界可能产生的一些微小偏差。
                # 如果上一个segment的结束时间比当前segment的开始时间还要晚，说明有重叠，
                # 且上一个segment包含了当前segment的开始部分。
                # 这种情况下，可以考虑调整上一个的结束，或当前segment的开始和文本。
                # 为简单起见，暂时直接添加，相信之前的过滤已处理主要重叠。
                if curr_s["start"] < prev_s["end"] and prev_s["text"].endswith(curr_s["text"][:len(prev_s["text"]) - int((prev_s["end"] - curr_s["start"])*10) ]): # 粗略检查
                     # 如果curr_s的开始部分被prev_s覆盖，并且文本也对应，则调整curr_s
                     # pass # 暂时不处理这种细微重叠，依赖模型切分
                     cleaned_segments.append(curr_s) # 仍添加，依赖后续文本拼接
                else:
                    cleaned_segments.append(curr_s)
        
        final_text = " ".join([s["text"] for s in cleaned_segments]).strip()
        language = self._detect_language(final_text)
        
        logger.info(f"分块转录完成。最终文本长度: {len(final_text)}, 分段数: {len(cleaned_segments)}")
        
        return TranscriptionResult(
            text=final_text,
            segments=cleaned_segments,
            language=language
        )
            
    def _perform_transcription(self, audio_data):
        """
        执行转录（需要在子类中实现）
        
        参数:
            audio_data: 音频数据（numpy数组）
            
        返回:
            模型的转录结果
        """
        raise NotImplementedError("子类必须实现_perform_transcription方法")
    
    def _get_text_from_result(self, result):
        """
        从结果中获取文本（需要在子类中实现）
        
        参数:
            result: 模型的转录结果
            
        返回:
            转录的文本
        """
        raise NotImplementedError("子类必须实现_get_text_from_result方法") 
"""
整合ASR和说话人分离的转录器模块，支持流式处理长语音对话
"""

import os
import numpy as np
from pydub import AudioSegment
from typing import Dict, List, Union, Optional, Tuple, Iterator, Any
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import re # 新增导入

# 导入ASR和说话人分离模块，使用相对导入
from .asr.asr_parakeet_mlx import MLXParakeetTranscriber, TranscriptionResult
from .diarization.diarization_pyannote import PyannoteTranscriber, DiarizationResult
from .schemas import EnhancedSegment, CombinedTranscriptionResult # 新增导入

# 配置日志
logger = logging.getLogger("podcast_transcribe")

class CombinedTranscriber:
    """整合ASR和说话人分离的转录器"""
    
    def __init__(
        self,
        asr_model_name: str = "mlx-community/parakeet-tdt-0.6b-v2",
        diarization_model_name: str = "pyannote/speaker-diarization-3.1",
        hf_token: Optional[str] = None,
        device: str = "cpu",
        segmentation_batch_size: int = 64,
        parallel: bool = False,
    ):
        """
        初始化转录器
        
        参数:
            asr_model_name: ASR模型名称
            diarization_model_name: 说话人分离模型名称
            hf_token: Hugging Face令牌
            device: 推理设备，'cpu'或'cuda'
            segmentation_batch_size: 分割批处理大小，默认为64
            parallel: 是否并行执行ASR和说话人分离，默认为False
        """
        self.asr_model_name = asr_model_name
        self.diarization_model_name = diarization_model_name
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.device = device
        self.segmentation_batch_size = segmentation_batch_size
        self.parallel = parallel
        
        logger.info(f"初始化组合转录器，ASR模型: {asr_model_name}，分离模型: {diarization_model_name}，分割批处理大小: {segmentation_batch_size}，并行执行: {parallel}")
        
        self.asr_model = None
        self.diarization_model = None
    
    def _load_asr_model(self):
        """按需加载ASR模型"""
        if self.asr_model is None:
            logger.info("加载ASR模型")
            self.asr_model = MLXParakeetTranscriber(model_name=self.asr_model_name)

    def _load_diarization_model(self):
        """按需加载说话人分离模型"""
        if self.diarization_model is None:
            logger.info("加载说话人分离模型")
            self.diarization_model = PyannoteTranscriber(
                model_name=self.diarization_model_name,
                token=self.hf_token,
                device=self.device,
                segmentation_batch_size=self.segmentation_batch_size
            )
    
    def _merge_adjacent_text_segments(self, segments: List[EnhancedSegment]) -> List[EnhancedSegment]:
        """
        合并相邻的、可能属于同一句子的 EnhancedSegment。
        合并条件：同一说话人，时间基本连续，文本内容可拼接。
        """
        if not segments:
            return []

        merged_segments: List[EnhancedSegment] = []
        if not segments: # 重复检查，可移除
            return merged_segments

        current_merged_segment = segments[0]

        for i in range(1, len(segments)):
            next_segment = segments[i]
            
            time_gap_seconds = next_segment.start - current_merged_segment.end
            
            can_merge_text = False
            if current_merged_segment.text and next_segment.text:
                current_text_stripped = current_merged_segment.text.strip()
                if current_text_stripped and not current_text_stripped[-1] in ".。?!？！":
                    can_merge_text = True
            
            if (current_merged_segment.speaker == next_segment.speaker and
                0 <= time_gap_seconds < 0.75 and 
                can_merge_text):
                current_merged_segment = EnhancedSegment(
                    start=current_merged_segment.start,
                    end=next_segment.end,
                    text=(current_merged_segment.text.strip() + " " + next_segment.text.strip()).strip(),
                    speaker=current_merged_segment.speaker,
                    language=current_merged_segment.language
                )
            else:
                merged_segments.append(current_merged_segment)
                current_merged_segment = next_segment
        
        merged_segments.append(current_merged_segment)
        
        return merged_segments

    def _run_asr(self, audio: AudioSegment) -> TranscriptionResult:
        """执行ASR处理"""
        self._load_asr_model()
        logger.debug("执行ASR...")
        return self.asr_model.transcribe(audio)
        
    def _run_diarization(self, audio: AudioSegment) -> DiarizationResult:
        """执行说话人分离处理"""
        self._load_diarization_model()
        logger.debug("执行说话人分离...")
        return self.diarization_model.diarize(audio)

    def transcribe(self, audio: AudioSegment) -> CombinedTranscriptionResult:
        """
        转录整个音频 (新的非流式逻辑将在这里实现)
        
        参数:
            audio: 要转录的AudioSegment对象
            
        返回:
            包含完整转录和说话人信息的结果
        """
        logger.info(f"开始转录 {len(audio)/1000:.2f} 秒的音频 (非流式)")

        if self.parallel:
            # 并行执行ASR和说话人分离
            logger.info("并行执行ASR和说话人分离")
            with ThreadPoolExecutor(max_workers=2) as executor:
                asr_future = executor.submit(self._run_asr, audio)
                diarization_future = executor.submit(self._run_diarization, audio)
                
                asr_result: TranscriptionResult = asr_future.result()
                diarization_result: DiarizationResult = diarization_future.result()
                
            logger.debug(f"ASR完成，识别语言: {asr_result.language}，得到 {len(asr_result.segments)} 个分段")
            logger.debug(f"说话人分离完成，得到 {len(diarization_result.segments)} 个说话人分段，检测到 {diarization_result.num_speakers} 个说话人")
        else:
            # 顺序执行ASR和说话人分离
            # 步骤1: 对整个音频执行ASR
            self._load_asr_model() # 按需加载ASR模型
            logger.debug("执行ASR...")
            asr_result: TranscriptionResult = self.asr_model.transcribe(audio)
            logger.debug(f"ASR完成，识别语言: {asr_result.language}，得到 {len(asr_result.segments)} 个分段")

            # 步骤2: 对整个音频执行说话人分离
            self._load_diarization_model() # 按需加载说话人分离模型
            logger.debug("执行说话人分离...")
            diarization_result: DiarizationResult = self.diarization_model.diarize(audio)
            logger.debug(f"说话人分离完成，得到 {len(diarization_result.segments)} 个说话人分段，检测到 {diarization_result.num_speakers} 个说话人")
        
        # 步骤3: 创建增强分段
        all_enhanced_segments: List[EnhancedSegment] = self._create_enhanced_segments_with_splitting(
            asr_result.segments, 
            diarization_result.segments, 
            asr_result.language
        )
        
        # 步骤4: （可选）合并相邻的文本分段
        if all_enhanced_segments:
            logger.debug(f"合并前有 {len(all_enhanced_segments)} 个增强分段，尝试合并相邻分段...")
            final_segments = self._merge_adjacent_text_segments(all_enhanced_segments)
            logger.debug(f"合并后有 {len(final_segments)} 个增强分段")
        else:
            final_segments = []
            logger.debug("没有增强分段可供合并。")

        # 整理合并的文本
        full_text = " ".join([segment.text for segment in final_segments]).strip()
        
        # 计算最终说话人数
        num_speakers_set = set(s.speaker for s in final_segments if s.speaker != "UNKNOWN")

        return CombinedTranscriptionResult(
            segments=final_segments,
            text=full_text,
            language=asr_result.language or "unknown",
            num_speakers=len(num_speakers_set) if num_speakers_set else diarization_result.num_speakers
        )

    # 新方法：根据标点分割ASR文本片段
    def _split_asr_segment_by_punctuation(
        self,
        asr_seg_text: str,
        asr_seg_start: float,
        asr_seg_end: float
    ) -> List[Dict[str, Any]]:
        """
        根据标点符号分割ASR文本片段，并按字符比例估算子片段的时间戳。
        返回: 字典列表，每个字典包含 'text', 'start', 'end'。
        """
        sentence_terminators = ".。?!？！;；"
        # 正则表达式：匹配句子内容以及紧随其后的标点（如果存在）
        # 使用 re.split 保留分隔符，然后重组
        parts = re.split(f'([{sentence_terminators}])', asr_seg_text)

        sub_texts_final = []
        current_s = ""
        for s_part in parts:
            if not s_part:
                continue
            current_s += s_part
            if s_part in sentence_terminators:
                if current_s.strip():
                    sub_texts_final.append(current_s.strip())
                current_s = ""
        if current_s.strip():
            sub_texts_final.append(current_s.strip())

        if not sub_texts_final or (len(sub_texts_final) == 1 and sub_texts_final[0] == asr_seg_text.strip()):
            # 没有有效分割或分割后只有一个句子（等于原始文本）
            return [{"text": asr_seg_text.strip(), "start": asr_seg_start, "end": asr_seg_end}]

        output_sub_segments = []
        total_text_len = len(asr_seg_text) # 使用原始文本长度进行比例计算
        if total_text_len == 0:
             return [{"text": "", "start": asr_seg_start, "end": asr_seg_end}]

        current_time = asr_seg_start
        original_duration = asr_seg_end - asr_seg_start

        for i, sub_text in enumerate(sub_texts_final):
            sub_len = len(sub_text)
            sub_duration = (sub_len / total_text_len) * original_duration
            
            sub_start_time = current_time
            sub_end_time = current_time + sub_duration
            
            # 对于最后一个分片，确保其结束时间与原始分段的结束时间一致，以避免累积误差
            if i == len(sub_texts_final) - 1:
                sub_end_time = asr_seg_end
            
            # 确保结束时间不超过原始结束时间，并且开始时间不晚于结束时间
            sub_end_time = min(sub_end_time, asr_seg_end)
            if sub_start_time >= sub_end_time and sub_start_time == asr_seg_end : # 如果开始等于原始结束，允许微小片段
                 if sub_text: # 仅当有文本时
                    output_sub_segments.append({"text": sub_text, "start": sub_start_time, "end": sub_end_time})
            elif sub_start_time < sub_end_time :
                output_sub_segments.append({"text": sub_text, "start": sub_start_time, "end": sub_end_time})

            current_time = sub_end_time
            if current_time >= asr_seg_end and i < len(sub_texts_final) -1: # 如果时间已用完，但还有句子
                # 将剩余句子附加到最后一个有效的时间段，或创建零长度的段
                logger.warning(f"时间已在分割过程中用尽，但仍有文本未分配时间。原始段: [{asr_seg_start}-{asr_seg_end}], 当前子句: '{sub_text}'")
                # 为后续未分配时间的文本创建零时长或极短时长的片段，附着在末尾
                for k in range(i + 1, len(sub_texts_final)):
                    remaining_text = sub_texts_final[k]
                    if remaining_text:
                         output_sub_segments.append({"text": remaining_text, "start": asr_seg_end, "end": asr_seg_end})
                break


        # 如果处理后没有任何子分段（例如原始文本为空，或分割逻辑问题），返回原始信息作为一个分段
        if not output_sub_segments and asr_seg_text.strip():
            return [{"text": asr_seg_text.strip(), "start": asr_seg_start, "end": asr_seg_end}]
        elif not output_sub_segments and not asr_seg_text.strip():
             return [{"text": "", "start": asr_seg_start, "end": asr_seg_end}]


        return output_sub_segments

    # 新的核心方法：创建增强分段，包含说话人分配和按需分裂逻辑
    def _create_enhanced_segments_with_splitting(
        self,
        asr_segments: List[Dict[str, Union[float, str]]],
        diarization_segments: List[Dict[str, Union[float, str, int]]],
        language: str
    ) -> List[EnhancedSegment]:
        """
        为ASR分段分配说话人，如果ASR分段跨越多个说话人，则尝试按标点分裂。
        """
        final_enhanced_segments: List[EnhancedSegment] = []

        if not asr_segments:
            return []
        
        # 为了快速查找，可以预处理 diarization_segments，但对于数量不多的情况，直接遍历也可
        # diarization_segments.sort(key=lambda x: x['start']) # 确保有序

        for asr_seg in asr_segments:
            asr_start = float(asr_seg["start"])
            asr_end = float(asr_seg["end"])
            asr_text = str(asr_seg["text"]).strip()

            if not asr_text or asr_start >= asr_end: # 跳过无效的ASR分段
                continue

            # 找出与当前ASR分段在时间上重叠的所有说话人分段
            overlapping_diar_segs = []
            for diar_seg in diarization_segments:
                diar_start = float(diar_seg["start"])
                diar_end = float(diar_seg["end"])
                
                overlap_start = max(asr_start, diar_start)
                overlap_end = min(asr_end, diar_end)
                
                if overlap_end > overlap_start: # 有重叠
                    overlapping_diar_segs.append({
                        "speaker": str(diar_seg["speaker"]),
                        "start": diar_start,
                        "end": diar_end,
                        "overlap_duration": overlap_end - overlap_start
                    })
            
            distinct_speakers_in_overlap = set(d['speaker'] for d in overlapping_diar_segs)

            segments_to_process_further: List[Dict[str, Any]] = []

            if len(distinct_speakers_in_overlap) > 1:
                logger.debug(f"ASR段 [{asr_start:.2f}-{asr_end:.2f}] \"{asr_text[:50]}...\" 跨越 {len(distinct_speakers_in_overlap)} 个说话人。尝试按标点分裂。")
                # 跨多个说话人，尝试按标点分裂ASR segment
                sub_asr_segments_data = self._split_asr_segment_by_punctuation(
                    asr_text,
                    asr_start,
                    asr_end
                )
                if len(sub_asr_segments_data) > 1:
                     logger.debug(f"成功将ASR段分裂成 {len(sub_asr_segments_data)} 个子句。")
                segments_to_process_further.extend(sub_asr_segments_data)
            else:
                # 单一说话人或无说话人重叠（也视为单一处理单位）
                segments_to_process_further.append({"text": asr_text, "start": asr_start, "end": asr_end})

            # 为每个原始或分裂后的ASR（子）分段分配说话人
            for current_proc_seg_data in segments_to_process_further:
                proc_text = current_proc_seg_data["text"].strip()
                proc_start = current_proc_seg_data["start"]
                proc_end = current_proc_seg_data["end"]

                if not proc_text or proc_start >= proc_end: # 跳过无效的子分段
                    continue

                # 为当前处理的（可能是子）分段确定最佳说话人
                speaker_overlaps_for_proc_seg = {}
                for diar_seg_info in overlapping_diar_segs: # 使用之前计算的、与原始ASR段重叠的diar_segs
                    # 现在需要计算这个 diar_seg_info 与 proc_seg 的重叠
                    overlap_start = max(proc_start, diar_seg_info["start"])
                    overlap_end = min(proc_end, diar_seg_info["end"])
                    
                    if overlap_end > overlap_start:
                        overlap_duration = overlap_end - overlap_start
                        speaker = diar_seg_info["speaker"]
                        speaker_overlaps_for_proc_seg[speaker] = \
                            speaker_overlaps_for_proc_seg.get(speaker, 0) + overlap_duration
                
                best_speaker = "UNKNOWN"
                if speaker_overlaps_for_proc_seg:
                    best_speaker = max(speaker_overlaps_for_proc_seg.items(), key=lambda x: x[1])[0]
                elif overlapping_diar_segs: # 如果子分段本身没有重叠，但原始ASR段有
                    # 可以选择原始ASR段中占比最大的，或者最近的
                    # 为简化，如果子分段无直接重叠，也可能标记为UNKNOWN，或尝试找最近的
                    # 这里采用：如果子分段无直接重叠，但在原始ASR段中有说话人，则使用原始ASR段中重叠最长的
                    # （此逻辑分支效果待观察，更简单的是直接UNKNOWN）
                    # 此处简化：若子分段无重叠，则为UNKNOWN
                    pass # best_speaker 默认为 UNKNOWN

                # 如果 best_speaker 仍为 UNKNOWN，但原始ASR段只有一个说话者，则使用该说话者
                if best_speaker == "UNKNOWN" and len(distinct_speakers_in_overlap) == 1:
                    best_speaker = list(distinct_speakers_in_overlap)[0]
                elif best_speaker == "UNKNOWN" and not overlapping_diar_segs:
                    # 如果整个ASR段都没有任何说话人信息，则确实是UNKNOWN
                    pass


                final_enhanced_segments.append(
                    EnhancedSegment(
                        start=proc_start,
                        end=proc_end,
                        text=proc_text,
                        speaker=best_speaker,
                        language=language # 所有子分段继承原始ASR段的语言
                    )
                )
        
        # 对最终结果按开始时间排序
        final_enhanced_segments.sort(key=lambda seg: seg.start)
        return final_enhanced_segments


def transcribe_audio(
    audio_segment: AudioSegment,
    asr_model_name: str = "mlx-community/parakeet-tdt-0.6b-v2",
    diarization_model_name: str = "pyannote/speaker-diarization-3.1",
    hf_token: Optional[str] = None,
    device: str = "cpu",
    segmentation_batch_size: int = 64,
    parallel: bool = False,
) -> CombinedTranscriptionResult: # 返回类型固定为 CombinedTranscriptionResult
    """
    整合ASR和说话人分离的音频转录函数 (仅支持非流式)
    
    参数:
        audio_segment: 输入的AudioSegment对象
        asr_model_name: ASR模型名称
        diarization_model_name: 说话人分离模型名称
        hf_token: Hugging Face令牌
        device: 推理设备，'cpu'或'cuda'
        segmentation_batch_size: 分割批处理大小，默认为64
        parallel: 是否并行执行ASR和说话人分离，默认为False
        
    返回:
        完整转录结果
    """
    logger.info(f"调用transcribe_audio函数 (非流式)，音频长度: {len(audio_segment)/1000:.2f}秒")
    
    transcriber = CombinedTranscriber(
        asr_model_name=asr_model_name,
        diarization_model_name=diarization_model_name,
        hf_token=hf_token,
        device=device,
        segmentation_batch_size=segmentation_batch_size,
        parallel=parallel
    )
    
    # 直接调用 transcribe 方法
    return transcriber.transcribe(audio_segment)

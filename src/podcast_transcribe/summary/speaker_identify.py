from typing import List, Dict, Optional, Any
import json
import re
from dataclasses import dataclass, field
from datetime import datetime

from ..schemas import EnhancedSegment, PodcastChannel, PodcastEpisode
from ..llm import llm_router


class SpeakerIdentifier:
    """
    说话人识别器类，用于根据转录分段和播客元数据识别说话人的真实姓名或昵称
    """
    
    def __init__(self, llm_model_name: Optional[str] = None, llm_provider: str = "gemma-mlx"):
        """
        初始化说话人识别器
        
        参数:
            llm_model_name: LLM模型名称，如果为None则使用默认模型
            llm_provider: LLM提供者，默认为"gemma-mlx"
        """
        self.llm_model_name = llm_model_name
        self.llm_provider = llm_provider
    
    def _clean_html(self, html_string: Optional[str]) -> str:
        """
        简单地从字符串中移除HTML标签并清理多余空白。
        """
        if not html_string:
            return ""
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', ' ', html_string)
        # 替换HTML实体（简单版本，只处理常见几个）
        text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        # 移除多余的空白符
        text = re.sub(r'\\s+', ' ', text).strip()
        return text

    def _get_dialogue_samples(
        self,
        segments: List[EnhancedSegment], 
        max_samples_per_speaker: int = 3,  # 增加样本数量
        max_length_per_sample: int = 200   # 增加样本长度
    ) -> Dict[str, List[str]]:
        """
        为每个说话人提取对话样本。
        """
        speaker_dialogues: Dict[str, List[str]] = {}
        for segment in segments:
            speaker = segment.speaker
            if speaker == "UNKNOWN" or not segment.text.strip(): # 跳过未知说话人或空文本
                continue
            
            if speaker not in speaker_dialogues:
                speaker_dialogues[speaker] = []
            
            if len(speaker_dialogues[speaker]) < max_samples_per_speaker:
                text_sample = segment.text.strip()[:max_length_per_sample]
                if len(segment.text.strip()) > max_length_per_sample:
                    text_sample += "..."
                speaker_dialogues[speaker].append(text_sample)
        return speaker_dialogues

    def recognize_speaker_names(
        self,
        segments: List[EnhancedSegment],
        podcast_info: Optional[PodcastChannel],
        episode_info: Optional[PodcastEpisode],
        max_shownotes_length: int = 1500, 
        max_desc_length: int = 500
    ) -> Dict[str, str]:
        """
        使用LLM根据转录分段和播客/剧集元数据识别说话人的真实姓名或昵称。

        参数:
            segments: 转录后的 EnhancedSegment 列表。
            podcast_info: 包含播客元数据的 PodcastChannel 对象。
            episode_info: 包含单集播客元数据的 PodcastEpisode 对象。
            max_shownotes_length: 用于Prompt的 Shownotes 最大字符数。
            max_desc_length: 用于Prompt的播客描述最大字符数。

        返回:
            一个字典，键是原始的 "SPEAKER_XX"，值是识别出的说话人名称。
        """
        unique_speaker_ids = sorted(list(set(seg.speaker for seg in segments if seg.speaker != "UNKNOWN" and seg.text.strip())))
        if not unique_speaker_ids:
            print("未能从 segments 中提取到有效的 speaker_ids。")
            return {}

        dialogue_samples = self._get_dialogue_samples(segments)
        
        # 增加每个说话人的话语分析信息，包括话语频率和长度
        speaker_stats = {}
        for segment in segments:
            speaker = segment.speaker
            if speaker == "UNKNOWN" or not segment.text.strip():
                continue
                
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    "total_segments": 0,
                    "total_chars": 0,
                    "avg_segment_length": 0,
                    "intro_likely": False  # 是否有介绍性质的话语
                }
            
            speaker_stats[speaker]["total_segments"] += 1
            speaker_stats[speaker]["total_chars"] += len(segment.text)
            
            # 检测可能的自我介绍或他人介绍
            lower_text = segment.text.lower()
            intro_patterns = [
                r'欢迎来到', r'欢迎收听', r'我是', r'我叫', r'大家好', r'今天的嘉宾是', r'我们请到了',
                r'welcome to', r'i\'m your host', r'this is', r'today we have', r'joining us', 
                r'our guest', r'my name is'
            ]
            if any(re.search(pattern, lower_text) for pattern in intro_patterns):
                speaker_stats[speaker]["intro_likely"] = True
        
        # 计算平均话语长度
        for speaker, stats in speaker_stats.items():
            if stats["total_segments"] > 0:
                stats["avg_segment_length"] = stats["total_chars"] / stats["total_segments"]
        
        # 创建增强的说话人信息，包含统计数据
        speaker_info_for_prompt = []
        for speaker_id in unique_speaker_ids:
            samples = dialogue_samples.get(speaker_id, ["(No dialogue samples available)"])
            stats = speaker_stats.get(speaker_id, {"total_segments": 0, "avg_segment_length": 0, "intro_likely": False})
            
            speaker_info_for_prompt.append({
                "speaker_id": speaker_id,
                "dialogue_samples": samples,
                "speech_stats": {
                    "total_segments": stats["total_segments"],
                    "avg_segment_length": round(stats["avg_segment_length"], 2),
                    "has_intro_pattern": stats["intro_likely"]
                }
            })

        # 安全地访问属性，提供默认值
        podcast_title = podcast_info.title if podcast_info and podcast_info.title else "Unknown Podcast"
        podcast_author = podcast_info.author if podcast_info and podcast_info.author else "Unknown"
        
        raw_podcast_desc = podcast_info.description if podcast_info and podcast_info.description else ""
        cleaned_podcast_desc = self._clean_html(raw_podcast_desc)
        podcast_desc_for_prompt = cleaned_podcast_desc[:max_desc_length]
        if len(cleaned_podcast_desc) > max_desc_length:
            podcast_desc_for_prompt += "..."
        
        episode_title = episode_info.title if episode_info and episode_info.title else "Unknown Episode"
        
        raw_episode_summary = episode_info.summary if episode_info and episode_info.summary else ""
        cleaned_episode_summary = self._clean_html(raw_episode_summary)
        episode_summary_for_prompt = cleaned_episode_summary[:max_desc_length] # 使用与描述相同的长度限制
        if len(cleaned_episode_summary) > max_desc_length:
            episode_summary_for_prompt += "..."

        raw_episode_shownotes = episode_info.shownotes if episode_info and episode_info.shownotes else ""
        cleaned_episode_shownotes = self._clean_html(raw_episode_shownotes)
        episode_shownotes_for_prompt = cleaned_episode_shownotes[:max_shownotes_length]
        if len(cleaned_episode_shownotes) > max_shownotes_length:
            episode_shownotes_for_prompt += "..."

        system_prompt = """You are an experienced podcast content analyst. Your task is to accurately identify the real names, nicknames, or roles of different speakers (tagged in SPEAKER_XX format) in a podcast episode, based on the provided metadata, episode information, dialogue snippets, and speech patterns. Your analysis should NOT rely on the order of speakers or speaker IDs."""

        user_prompt_template = f"""
Contextual Information:

1.  **Podcast Information**:
    *   Podcast Title: {podcast_title}
    *   Podcast Author/Producer: {podcast_author} (This information often points to the main host or production team)
    *   Podcast Description: {podcast_desc_for_prompt}

2.  **Current Episode Information**:
    *   Episode Title: {episode_title}
    *   Episode Summary: {episode_summary_for_prompt}
    *   Detailed Episode Notes (Shownotes):
        ```text
        {episode_shownotes_for_prompt}
        ```
        (Pay close attention to any host names, guest names, positions, or social media handles mentioned in the Shownotes.)

3.  **Speakers to Identify and Their Information**:
    ```json
    {json.dumps(speaker_info_for_prompt, ensure_ascii=False, indent=2)}
    ```
    (Analyze dialogue samples and speech statistics to understand speaker roles and identities. DO NOT use speaker IDs to determine roles - SPEAKER_00 is not necessarily the host.)

Task:
Based on all the information above, assign the most accurate name or role to each "speaker_id".

Analysis Guidance:
* A host typically has more frequent, shorter segments, often introduces the show or guests, and may mention the podcast name
* In panel discussion formats, there might be multiple hosts or co-hosts of similar speaking patterns
* In interview formats, the host typically asks questions while guests give longer answers
* Speakers who make introductory statements or welcome listeners are likely hosts
* Use dialogue content (not just speaking patterns) to identify names and roles

Output Requirements and Guidelines:
*   Please return the result strictly in JSON format. The keys of the JSON object should be the original "speaker_id" (e.g., "SPEAKER_00"), and the values should be the identified person's name or role (string type).
*   **Prioritize Specific Names/Nicknames**: If there is sufficient information (e.g., guests explicitly listed in Shownotes, or names mentioned in dialogue), please use the identified specific names, such as "John Doe", "AI Assistant", "Dr. Evelyn Reed". Do NOT append roles like "(Host)" or "(Guest)" if a specific name is found.
*   **Host Identification**:
    *   Hosts may be identified by analyzing speech patterns - they often speak more frequently in shorter segments
    *   Look for introduction patterns in dialogue where speakers welcome listeners or introduce the show
    *   The podcast author (if provided and credible) is often a host but verify through dialogue
    *   There may be multiple hosts (co-hosts) in panel-style podcasts
    *   If a host's name is identified, use the identified name directly (e.g., "Lex Fridman"). Do not append "(Host)".
    *   If the host's name cannot be determined but the role is clearly a host, use "Podcast Host".
*   **Guest Identification**:
    *   Guests often give longer responses and speak less frequently than hosts
    *   For other non-host speakers, if a specific name is identified, use the identified name directly (e.g., "John Carmack"). Do not append "(Guest)".
    *   If specific names cannot be identified for guests, label them sequentially as "Guest 1", "Guest 2", etc.
*   **Handling Multiple Hosts/Guests**: If there are multiple hosts or guests and they can be distinguished by name, use their names. If you cannot distinguish specific identities but know there are multiple hosts, use "Host 1", "Host 2", etc. Similarly for guests without specific names, use "Guest 1", "Guest 2".
*   **Ensure Completeness**: The returned JSON object must include all "speaker_id"s listed in the input as keys.

JSON Output Example:
```json
{{
  "SPEAKER_00": "Jane Smith",
  "SPEAKER_01": "Podcast Host",
  "SPEAKER_02": "Alex Green"
}}
```
Note that in this example, SPEAKER_01 is identified as the host, not SPEAKER_00, based on content analysis, not ID order.

Please begin your analysis and provide the JSON result.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_template}
        ]

        # 预设默认映射，使用更智能的启发式方法而不是简单依赖顺序
        final_map = {}
        
        # 尝试使用说话模式启发式方法来初步识别角色
        # 1. 说话次数最多的可能是主持人
        # 2. 有介绍性话语的可能是主持人
        # 3. 其他角色先标记为嘉宾
        
        host_candidates = []
        for speaker_id, stats in speaker_stats.items():
            if stats["intro_likely"]:
                host_candidates.append((speaker_id, 2))  # 优先级2：有介绍性话语
            else:
                # 按说话次数排序
                host_candidates.append((speaker_id, stats["total_segments"]))
        
        # 按可能性排序（介绍性话语 > 说话次数）
        host_candidates.sort(key=lambda x: (-1 if x[1] == 2 else 0, x[1]), reverse=True)
        
        if host_candidates:
            # 最可能的主持人
            host_id = host_candidates[0][0]
            final_map[host_id] = "Podcast Host"
            
            # 其他人先标为嘉宾
            guest_counter = 1
            for speaker_id in unique_speaker_ids:
                if speaker_id != host_id:
                    final_map[speaker_id] = f"Guest {guest_counter}"
                    guest_counter += 1
        else:
            # 如果没有明显线索，使用传统的顺序方法作为备选
            is_host_assigned = False
            guest_counter = 1
            for speaker_id in unique_speaker_ids:
                if not is_host_assigned:
                    final_map[speaker_id] = "Podcast Host"
                    is_host_assigned = True
                else:
                    final_map[speaker_id] = f"Guest {guest_counter}"
                    guest_counter += 1
            
        try:
            response = llm_router.chat_completion(
                messages=messages, 
                provider=self.llm_provider,
                model=self.llm_model_name,
                temperature=0.1, 
                max_tokens=1024
            )
            assistant_response_content = response["choices"][0]["message"]["content"]
            
            parsed_llm_output = None
            # 尝试从Markdown代码块中提取JSON
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', assistant_response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 如果没有markdown块，尝试找到第一个 '{' 到最后一个 '}'
                first_brace = assistant_response_content.find('{')
                last_brace = assistant_response_content.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_str = assistant_response_content[first_brace : last_brace+1]
                else: # 如果还是找不到，就认为整个回复都是JSON（可能需要更复杂的清理）
                    json_str = assistant_response_content.strip()
            
            try:
                parsed_llm_output = json.loads(json_str)
                if not isinstance(parsed_llm_output, dict): # 确保解析出来是字典
                    print(f"LLM返回的JSON不是一个字典: {parsed_llm_output}")
                    parsed_llm_output = None # 重置，以便使用默认值
            except json.JSONDecodeError as e:
                print(f"LLM返回的JSON解析失败: {e}")
                print(f"用于解析的字符串: '{json_str}'")
                # parsed_llm_output 保持为 None，将使用默认值

            if parsed_llm_output:
                # 直接使用LLM的有效输出，不再依赖预设的角色分配逻辑
                final_map = {}
                unknown_counter = 1
                
                # 先处理LLM识别出的角色
                for spk_id in unique_speaker_ids:
                    if spk_id in parsed_llm_output and isinstance(parsed_llm_output[spk_id], str) and parsed_llm_output[spk_id].strip():
                        final_map[spk_id] = parsed_llm_output[spk_id].strip()
                    else:
                        # 如果LLM没有给出特定ID的结果，使用"Unknown Speaker"
                        final_map[spk_id] = f"Unknown Speaker {unknown_counter}"
                        unknown_counter += 1
                
                # 检查是否有"Host"或"主持人"标识
                has_host = any("主持人" in name or "Host" in name for name in final_map.values())
                
                # 如果没有任何主持人标识，且存在"Unknown Speaker"，可以考虑将最活跃的未知说话人设为主持人
                if not has_host and any("Unknown Speaker" in name for name in final_map.values()):
                    # 找出最活跃的未知说话人
                    most_active_unknown = None
                    max_segments = 0
                    
                    for spk_id, name in final_map.items():
                        if "Unknown Speaker" in name and spk_id in speaker_stats:
                            if speaker_stats[spk_id]["total_segments"] > max_segments:
                                max_segments = speaker_stats[spk_id]["total_segments"]
                                most_active_unknown = spk_id
                    
                    if most_active_unknown:
                        final_map[most_active_unknown] = "Podcast Host"
            
            return final_map

        except Exception as e:
            import traceback
            print(f"调用LLM或处理响应时发生严重错误: {e}")
            print(traceback.format_exc())
            # 发生任何严重错误，返回初始的启发式映射
            return final_map
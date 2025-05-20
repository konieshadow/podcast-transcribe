from typing import List, Dict, Optional, Any
import json
import re
from dataclasses import dataclass, field
from datetime import datetime

# Actual imports from the project structure
from ..schemas import EnhancedSegment, PodcastChannel, PodcastEpisode
from ..llm.llm_gemma import GemmaMLXChatCompletion

# # --- Mock classes for standalone testing (comment out or remove in production) ---
# @dataclass
# class EnhancedSegment:
#     start: float
#     end: float
#     text: str
#     speaker: str
#     language: str

# @dataclass
# class PodcastEpisode:
#     title: Optional[str] = None
#     link: Optional[str] = None
#     published_date: Optional[datetime] = None
#     summary: Optional[str] = None
#     shownotes: Optional[str] = None
#     audio_url: Optional[str] = None
#     guid: Optional[str] = None
#     duration: Optional[str] = None
#     episode_type: Optional[str] = None
#     season: Optional[int] = None
#     episode_number: Optional[int] = None

# @dataclass
# class PodcastChannel:
#     title: Optional[str] = None
#     link: Optional[str] = None
#     description: Optional[str] = None
#     language: Optional[str] = None
#     image_url: Optional[str] = None
#     author: Optional[str] = None
#     last_build_date: Optional[datetime] = None
#     episodes: List[PodcastEpisode] = field(default_factory=list)

# class MockGemmaMLXChatCompletion:
#     def __init__(self, model_name: str = "mock_gemma"):
#         self.model_name = model_name
#         print(f"MockGemmaMLXChatCompletion initialized with model: {model_name}")

#     def create(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int, **kwargs) -> Dict:
#         print("---- Mock LLM Input Messages ----")
#         # for msg in messages:
#         #     print(f"Role: {msg['role']}")
#         #     print(f"Content: {msg['content']}")
#         #     print("-" * 20)
        
#         prompt_content = ""
#         for msg in messages:
#             if msg['role'] == 'user':
#                 prompt_content += msg['content']

#         # Try to find speaker_ids from the prompt to make the mock more dynamic
#         speaker_ids_in_prompt = []
#         try:
#             # Assuming the speaker info is a JSON string within the user prompt
#             speaker_data_match = re.search(r'(\\[\\s*\\{.*?speaker_id.*?\\}\\s*\\])', prompt_content, re.DOTALL)
#             if speaker_data_match:
#                 speaker_data_str = speaker_data_match.group(1)
#                 # Correcting the regex for extracting speaker_id values
#                 loaded_speaker_data = json.loads(speaker_data_str)
#                 for item in loaded_speaker_data:
#                     speaker_ids_in_prompt.append(item["speaker_id"])
#             else: # Fallback if JSON block not found, try simple regex (less robust)
#                 speaker_ids_in_prompt = re.findall(r'"speaker_id":\\s*"(SPEAKER_\\d+)"', prompt_content)
#         except Exception as e:
#             print(f"Mock LLM: Error parsing speaker_ids from prompt: {e}")
#             # Fallback to predefined if parsing fails
#             speaker_ids_in_prompt = ["SPEAKER_00", "SPEAKER_01"]


#         mock_response_content = {}
#         if not speaker_ids_in_prompt: # Ensure there's at least one speaker for the mock
#             speaker_ids_in_prompt.append("SPEAKER_00")

#         if speaker_ids_in_prompt[0] not in mock_response_content: # Assign first as host
#             mock_response_content[speaker_ids_in_prompt[0]] = "播客主持 Mock"
        
#         guest_idx = 1
#         for spk_id in speaker_ids_in_prompt[1:]:
#             if spk_id not in mock_response_content:
#                 mock_response_content[spk_id] = f"嘉宾 {guest_idx} Mock"
#                 guest_idx += 1
        
#         print(f"Mock LLM will respond with: {mock_response_content}")

#         return {
#             "choices": [
#                 {
#                     "message": {
#                         "role": "assistant",
#                         "content": json.dumps(mock_response_content, ensure_ascii=False)
#                     }
#                 }
#             ],
#             "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
#         }
# # --- End of Mock classes ---


def _clean_html(html_string: Optional[str]) -> str:
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
    segments: List[EnhancedSegment], 
    max_samples_per_speaker: int = 2, 
    max_length_per_sample: int = 150 # 增加样本长度
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
    segments: List[EnhancedSegment],
    podcast_info: Optional[PodcastChannel],
    episode_info: Optional[PodcastEpisode],
    llm_client: GemmaMLXChatCompletion, 
    max_shownotes_length: int = 1500, 
    max_desc_length: int = 500
) -> Dict[str, str]:
    """
    使用LLM根据转录分段和播客/剧集元数据识别说话人的真实姓名或昵称。

    参数:
        segments: 转录后的 EnhancedSegment 列表。
        podcast_info: 包含播客元数据的 PodcastChannel 对象。
        episode_info: 包含单集播客元数据的 PodcastEpisode 对象。
        llm_client: GemmaMLXChatCompletion 的实例。
        max_shownotes_length: 用于Prompt的 Shownotes 最大字符数。
        max_desc_length: 用于Prompt的播客描述最大字符数。

    返回:
        一个字典，键是原始的 "SPEAKER_XX"，值是识别出的说话人名称。
    """
    unique_speaker_ids = sorted(list(set(seg.speaker for seg in segments if seg.speaker != "UNKNOWN" and seg.text.strip())))
    if not unique_speaker_ids:
        print("未能从 segments 中提取到有效的 speaker_ids。")
        return {}

    dialogue_samples = _get_dialogue_samples(segments)
    
    speaker_info_for_prompt = []
    for speaker_id in unique_speaker_ids:
        samples = dialogue_samples.get(speaker_id, ["(No dialogue samples available)"])
        # 确保即使只有一个样本，也用 " | " 连接（虽然这里只会有一个）
        speaker_info_for_prompt.append({
            "speaker_id": speaker_id,
            "dialogue_sample": " | ".join(samples) 
        })

    # 安全地访问属性，提供默认值
    podcast_title = podcast_info.title if podcast_info and podcast_info.title else "Unknown Podcast"
    podcast_author = podcast_info.author if podcast_info and podcast_info.author else "Unknown"
    
    raw_podcast_desc = podcast_info.description if podcast_info and podcast_info.description else ""
    cleaned_podcast_desc = _clean_html(raw_podcast_desc)
    podcast_desc_for_prompt = cleaned_podcast_desc[:max_desc_length]
    if len(cleaned_podcast_desc) > max_desc_length:
        podcast_desc_for_prompt += "..."
    
    episode_title = episode_info.title if episode_info and episode_info.title else "Unknown Episode"
    
    raw_episode_summary = episode_info.summary if episode_info and episode_info.summary else ""
    cleaned_episode_summary = _clean_html(raw_episode_summary)
    episode_summary_for_prompt = cleaned_episode_summary[:max_desc_length] # 使用与描述相同的长度限制
    if len(cleaned_episode_summary) > max_desc_length:
        episode_summary_for_prompt += "..."

    raw_episode_shownotes = episode_info.shownotes if episode_info and episode_info.shownotes else ""
    cleaned_episode_shownotes = _clean_html(raw_episode_shownotes)
    episode_shownotes_for_prompt = cleaned_episode_shownotes[:max_shownotes_length]
    if len(cleaned_episode_shownotes) > max_shownotes_length:
        episode_shownotes_for_prompt += "..."

    system_prompt = """You are an experienced podcast content analyst. Your task is to accurately identify the real names, nicknames, or roles of different speakers (tagged in SPEAKER_XX format) in a podcast episode, based on the provided metadata, episode information, and dialogue snippets."""

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

3.  **Speakers to Identify and Their Dialogue Samples**:
    ```json
    {json.dumps(speaker_info_for_prompt, ensure_ascii=False, indent=2)}
    ```
    (Dialogue samples may contain self-introductions, how speakers address each other, or discussions related to their identities.)

Task:
Based on all the information above, assign the most accurate name or role to each "speaker_id".

Output Requirements and Guidelines:
*   Please return the result strictly in JSON format. The keys of the JSON object should be the original "speaker_id" (e.g., "SPEAKER_00"), and the values should be the identified person's name or role (string type).
*   **Prioritize Specific Names/Nicknames**: If there is sufficient information (e.g., guests explicitly listed in Shownotes, or names mentioned in dialogue), please use the identified specific names, such as "John Doe", "AI Assistant", "Dr. Evelyn Reed".
*   **Host Identification**:
    *   The podcast author (if provided and credible) is often the host.
    *   Identify roles explicitly stated as "host", "presenter", etc., in the dialogue or Shownotes.
    *   If a host's name is identified, you can use formats like: "[Identified Name] (Host)" or "Host [Identified Name]".
    *   If the name cannot be determined but the role is clearly a host, use "Podcast Host".
*   **Guest Identification**:
    *   For other non-host speakers, if specific names cannot be identified, label them sequentially as "Guest 1", "Guest 2", etc.
    *   If a guest's name is identified, you can use formats like: "[Identified Name] (Guest)".
*   **Handling Multiple Hosts/Guests**: If there are multiple hosts or guests and they can be distinguished, name them accordingly. If you cannot distinguish specific identities but know there are multiple hosts, you can use "Host 1", "Host 2".
*   **Insufficient Information**: If there is too little information for a particular "speaker_id" to make an informed judgment, conservatively use "Unknown Speaker" or determine if they are likely "Podcast Host" or "Guest" based on context.
*   **Ensure Completeness**: The returned JSON object must include all "speaker_id"s listed in "Speakers to Identify and Their Dialogue Samples" as keys.

JSON Output Example:
```json
{{
  "SPEAKER_00": "John Doe (Host)",
  "SPEAKER_01": "Jane Smith (Guest)",
  "SPEAKER_02": "Alex Green (Guest 2)"
}}
```
Or, in cases with very limited information:
```json
{{
  "SPEAKER_00": "Podcast Host",
  "SPEAKER_01": "Guest 1"
}}
```
Please begin your analysis and provide the JSON result.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_template}
    ]

    final_map = {}
    # 预先填充默认值，以防LLM调用失败或返回不完整
    is_host_assigned_default = False
    guest_counter_default = 1
    for spk_id in unique_speaker_ids:
        if not is_host_assigned_default:
            final_map[spk_id] = "Podcast Host" # 默认第一个是主持人
            is_host_assigned_default = True
        else:
            final_map[spk_id] = f"Guest {guest_counter_default}"
            guest_counter_default += 1
            
    try:
        # print("---- Sending to LLM ----")
        # print(f"System Prompt: {system_prompt}")
        # print(f"User Prompt: {user_prompt_template}")
        # print("-------------------------")
        
        response = llm_client.create(messages=messages, temperature=0.1, max_tokens=1024) # 低温以获得更确定的JSON输出
        assistant_response_content = response["choices"][0]["message"]["content"]
        
        # print(f"---- LLM Raw Response ----\n{assistant_response_content}\n-------------------------")

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
            # 用LLM的有效输出来更新 final_map
            # 同时检查LLM是否包含了所有speaker_id
            temp_host_assigned = False
            temp_guest_counter = 1
            
            processed_ids_from_llm = set()

            for spk_id in unique_speaker_ids: # 遍历我们期望的ID
                if spk_id in parsed_llm_output and isinstance(parsed_llm_output[spk_id], str) and parsed_llm_output[spk_id].strip():
                    final_map[spk_id] = parsed_llm_output[spk_id].strip()
                    processed_ids_from_llm.add(spk_id)
                    if "主持人" in final_map[spk_id] or "Host" in final_map[spk_id]:
                        temp_host_assigned = True
                # else: # 如果LLM输出中缺少某个ID，或值无效，则保留final_map中的默认值

            # 对于LLM输出中存在但我们未在unique_speaker_ids中跟踪的ID（不太可能发生，但为了健壮性）
            # 或者，如果LLM返回的ID少于unique_speaker_ids，则需要补充
            # 这里的逻辑是：以unique_speaker_ids为准，用LLM结果填充，不足的保留默认值

            # 重新检查并分配 "Host" 和 "Guest N" 如果LLM的命名不清晰
            # 这个阶段，final_map 已经混合了 LLM 的输出和一些初始默认值
            # 我们需要确保至少有一个Host，并且Guest编号正确
            
            current_names = list(final_map[spk_id] for spk_id in unique_speaker_ids) # 按原始顺序获取名称
            
            has_host_role = any("主持人" in name or "Host" in name for name in current_names)
            
            # 如果LLM的命名中没有明确的Host，但我们之前默认分配了一个，可能需要调整
            # 或者，如果LLM将所有人都命名了但没有Host，选择一个作为Host
            
            final_updated_map = {}
            guest_idx_refill = 1
            host_assigned_refill = False

            # 优先保留LLM识别出的具体名字和角色
            for spk_id in unique_speaker_ids:
                name_from_llm = final_map.get(spk_id) # 这是混合了LLM和初始默认的
                
                is_generic_guest = re.match(r"Guest \\d+", name_from_llm) or name_from_llm == "Guest"
                is_generic_host = name_from_llm == "Podcast Host"

                if "主持人" in name_from_llm or "Host" in name_from_llm:
                    final_updated_map[spk_id] = name_from_llm
                    host_assigned_refill = True
                elif not is_generic_guest and not is_generic_host and name_from_llm.strip() != "UNKNOWN": # 具体名字
                    final_updated_map[spk_id] = name_from_llm
                # 对于其他情况（LLM未提供，或提供了通用占位符），我们将重新分配
            
            # 重新分配那些没有被LLM赋予明确角色的
            for spk_id in unique_speaker_ids:
                if spk_id not in final_updated_map: # 如果还没被上面逻辑处理
                    if not host_assigned_refill:
                        final_updated_map[spk_id] = "Podcast Host"
                        host_assigned_refill = True
                    else:
                        final_updated_map[spk_id] = f"Guest {guest_idx_refill}"
                        guest_idx_refill +=1
            final_map = final_updated_map

        # print(f"---- Speaker Name Map (final) ----\n{json.dumps(final_map, ensure_ascii=False, indent=2)}\n-------------------------------")
        return final_map

    except Exception as e:
        import traceback
        print(f"调用LLM或处理响应时发生严重错误: {e}")
        print(traceback.format_exc())
        # 发生任何严重错误，返回初始的默认映射
        # final_map 此时应该已经是初始的默认映射
        return final_map

import gradio as gr
import requests
import io
from pydub import AudioSegment
import traceback # 用于打印更详细的错误信息
import tempfile
import os
import uuid
import atexit
import shutil
import threading
import time
from typing import Dict, Set

# 尝试相对导入，这在通过 `python -m src.podcast_transcribe.webui.app` 运行时有效
try:
    from podcast_transcribe.rss.podcast_rss_parser import parse_podcast_rss
    from podcast_transcribe.schemas import PodcastChannel, PodcastEpisode, CombinedTranscriptionResult, EnhancedSegment
    from podcast_transcribe.transcriber import transcribe_podcast_audio
except ImportError:
    # 如果直接运行此脚本，并且项目根目录不在PYTHONPATH中，
    # 则需要将项目根目录添加到 sys.path
    import sys
    import os
    # 获取当前脚本文件所在的目录 (src/podcast_transcribe/webui)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录 (向上三级: webui -> podcast_transcribe -> src -> project_root)
    # 修正：应该是 src 的父目录是项目根
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    # 将 src 目录添加到 sys.path，因为模块是 podcast_transcribe.xxx
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    from podcast_transcribe.rss.podcast_rss_parser import parse_podcast_rss
    from podcast_transcribe.schemas import PodcastChannel, PodcastEpisode, CombinedTranscriptionResult, EnhancedSegment
    from podcast_transcribe.transcriber import transcribe_podcast_audio

# 会话级别的临时文件管理
class SessionFileManager:
    def __init__(self):
        self.session_files: Dict[str, Set[str]] = {}
        self.lock = threading.Lock()
        self.cleanup_thread = None
        self.start_cleanup_thread()
    
    def start_cleanup_thread(self):
        """启动后台清理线程"""
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
            self.cleanup_thread.start()
    
    def _periodic_cleanup(self):
        """定期清理过期的临时文件"""
        while True:
            try:
                time.sleep(300)  # 每5分钟清理一次
                self._cleanup_old_files()
            except Exception as e:
                print(f"清理线程错误: {e}")
    
    def _cleanup_old_files(self):
        """清理超过30分钟的临时文件"""
        current_time = time.time()
        with self.lock:
            for session_id, files in list(self.session_files.items()):
                files_to_remove = []
                for filepath in list(files):
                    try:
                        if os.path.exists(filepath):
                            # 检查文件创建时间
                            file_age = current_time - os.path.getctime(filepath)
                            if file_age > 1800:  # 30分钟
                                os.remove(filepath)
                                files_to_remove.append(filepath)
                                print(f"自动清理过期临时文件: {filepath}")
                        else:
                            files_to_remove.append(filepath)
                    except Exception as e:
                        print(f"清理文件 {filepath} 时出错: {e}")
                        files_to_remove.append(filepath)
                
                # 从集合中移除已清理的文件
                for filepath in files_to_remove:
                    files.discard(filepath)
                
                # 如果会话没有文件了，移除会话记录
                if not files:
                    del self.session_files[session_id]
    
    def add_file(self, session_id: str, filepath: str):
        """添加文件到会话管理"""
        with self.lock:
            if session_id not in self.session_files:
                self.session_files[session_id] = set()
            self.session_files[session_id].add(filepath)
    
    def cleanup_session(self, session_id: str):
        """清理特定会话的所有文件"""
        with self.lock:
            if session_id in self.session_files:
                files = self.session_files[session_id]
                for filepath in list(files):
                    try:
                        if os.path.exists(filepath):
                            os.remove(filepath)
                            print(f"清理会话文件: {filepath}")
                    except Exception as e:
                        print(f"无法删除文件 {filepath}: {e}")
                del self.session_files[session_id]
    
    def cleanup_all(self):
        """清理所有临时文件"""
        with self.lock:
            total_files = 0
            for session_id in list(self.session_files.keys()):
                total_files += len(self.session_files[session_id])
                self.cleanup_session(session_id)
            print(f"应用程序退出，清理了 {total_files} 个临时文件")

# 全局文件管理器
file_manager = SessionFileManager()

def cleanup_temp_files():
    """清理应用程序使用的临时文件"""
    file_manager.cleanup_all()

# 注册应用程序退出时的清理函数
atexit.register(cleanup_temp_files)

def get_session_id(request: gr.Request = None) -> str:
    """获取会话ID，用于文件管理"""
    if request and hasattr(request, 'session_hash'):
        return request.session_hash
    else:
        # 如果无法获取会话ID，使用UUID
        return str(uuid.uuid4())

# 添加资源限制检查
def check_system_resources():
    """检查系统资源是否足够"""
    try:
        import psutil
        # 检查可用内存
        memory = psutil.virtual_memory()
        if memory.available < 500 * 1024 * 1024:  # 少于500MB
            return False, "系统内存不足，请稍后再试"
        
        # 检查磁盘空间
        disk = psutil.disk_usage(tempfile.gettempdir())
        if disk.free < 1024 * 1024 * 1024:  # 少于1GB
            return False, "磁盘空间不足，请稍后再试"
        
        return True, "资源充足"
    except ImportError:
        # 如果没有psutil，跳过检查
        return True, "无法检查资源状态"
    except Exception as e:
        print(f"资源检查错误: {e}")
        return True, "资源检查失败，继续执行"

def parse_rss_feed(rss_url: str):
    """回调函数：解析 RSS Feed"""
    print(f"开始解析RSS: {rss_url}")
    
    if not rss_url:
        print("RSS地址为空")
        return {
            status_message_area: gr.update(value="Error: Please enter an RSS URL."),
            podcast_title_display: gr.update(value="", visible=False),
            episode_dropdown: gr.update(choices=[], value=None, interactive=False),
            podcast_data_state: None,
            audio_player: gr.update(value=None),
            current_audio_url_state: None,
            episode_shownotes: gr.update(value="", visible=False),
            transcription_output_df: gr.update(value=None, headers=["Speaker", "Text", "Time"]),
            transcribe_button: gr.update(interactive=False),
            selected_episode_index_state: None
        }
    
    try:
        print(f"正在解析RSS: {rss_url}")
        # 先更新状态消息，但由于不再使用生成器，我们直接在解析后更新UI
        
        podcast_data: PodcastChannel = parse_podcast_rss(rss_url)
        print(f"RSS解析结果: 频道名称={podcast_data.title if podcast_data else 'None'}, 剧集数量={len(podcast_data.episodes) if podcast_data and podcast_data.episodes else 0}")
        
        if podcast_data and podcast_data.episodes:
            choices = []
            for i, episode in enumerate(podcast_data.episodes):
                # 使用 (标题 (时长), guid 或索引) 作为选项
                # 如果 guid 不可靠或缺失，可以使用索引
                label = f"{episode.title or 'Untitled'} (Duration: {episode.duration or 'Unknown'})"
                # 将 episode 对象直接作为值传递，或仅传递一个唯一标识符
                # 为了简单起见，我们使用索引作为唯一ID，因为我们需要从 podcast_data_state 中检索完整的 episode
                choices.append((label, i)) 
            
            # 显示播客标题
            podcast_title = f"## 🎙️ {podcast_data.title or 'Unknown Podcast'}"
            if podcast_data.author:
                podcast_title += f"\n**Host/Producer:** {podcast_data.author}"
            if podcast_data.description:
                # 限制描述长度，避免界面过长
                description = podcast_data.description[:300]
                if len(podcast_data.description) > 300:
                    description += "..."
                podcast_title += f"\n\n**Podcast Description:** {description}"
            
            return {
                status_message_area: gr.update(value=f"Successfully parsed {len(podcast_data.episodes)} episodes. Please select an episode."),
                podcast_title_display: gr.update(value=podcast_title, visible=True),
                episode_dropdown: gr.update(choices=choices, value=None, interactive=True),
                podcast_data_state: podcast_data,
                audio_player: gr.update(value=None),
                current_audio_url_state: None,
                episode_shownotes: gr.update(value="", visible=False),
                transcription_output_df: gr.update(value=None),
                transcribe_button: gr.update(interactive=False),
                selected_episode_index_state: None
            }
        elif podcast_data: # 有 channel 信息但没有 episodes
            print("解析成功但未找到剧集")
            podcast_title = f"## 🎙️ {podcast_data.title or 'Unknown Podcast'}"
            if podcast_data.author:
                podcast_title += f"\n**Host/Producer:** {podcast_data.author}"
            
            return {
                status_message_area: gr.update(value="Parsing successful, but no episodes found."),
                podcast_title_display: gr.update(value=podcast_title, visible=True),
                episode_dropdown: gr.update(choices=[], value=None, interactive=False),
                podcast_data_state: podcast_data, # 仍然存储，以防万一
                audio_player: gr.update(value=None),
                current_audio_url_state: None,
                episode_shownotes: gr.update(value="", visible=False),
                transcription_output_df: gr.update(value=None),
                transcribe_button: gr.update(interactive=False),
                selected_episode_index_state: None
            }
        else:
            print(f"解析RSS失败: {rss_url}")
            return {
                status_message_area: gr.update(value=f"Failed to parse RSS: {rss_url}. Please check the URL or network connection."),
                podcast_title_display: gr.update(value="", visible=False),
                episode_dropdown: gr.update(choices=[], value=None, interactive=False),
                podcast_data_state: None,
                audio_player: gr.update(value=None),
                current_audio_url_state: None,
                episode_shownotes: gr.update(value="", visible=False),
                transcription_output_df: gr.update(value=None),
                transcribe_button: gr.update(interactive=False),
                selected_episode_index_state: None
            }
    except Exception as e:
        print(f"解析 RSS 时发生错误: {e}")
        traceback.print_exc()
        return {
            status_message_area: gr.update(value=f"Serious error occurred while parsing RSS: {e}"),
            podcast_title_display: gr.update(value="", visible=False),
            episode_dropdown: gr.update(choices=[], value=None, interactive=False),
            podcast_data_state: None,
            audio_player: gr.update(value=None),
            current_audio_url_state: None,
            episode_shownotes: gr.update(value="", visible=False),
            transcription_output_df: gr.update(value=None),
            transcribe_button: gr.update(interactive=False),
            selected_episode_index_state: None
        }

def load_episode_audio(selected_episode_index: int, podcast_data: PodcastChannel, request: gr.Request = None):
    """回调函数：当用户从下拉菜单选择一个剧集时加载音频"""
    print(f"开始加载剧集音频，选择的索引: {selected_episode_index}")
    
    # 获取会话ID
    session_id = get_session_id(request)
    
    # 检查系统资源
    resource_ok, resource_msg = check_system_resources()
    if not resource_ok:
        return {
            audio_player: gr.update(value=None),
            current_audio_url_state: None,
            status_message_area: gr.update(value=f"⚠️ {resource_msg}"),
            episode_shownotes: gr.update(value="", visible=False),
            transcription_output_df: gr.update(value=None),
            local_audio_file_path: None,
            transcribe_button: gr.update(interactive=False),
            selected_episode_index_state: None
        }
    
    if selected_episode_index is None or podcast_data is None or not podcast_data.episodes:
        print("未选择剧集或无播客数据")
        return {
            audio_player: gr.update(value=None),
            current_audio_url_state: None,
            status_message_area: gr.update(value="Please parse RSS first and select an episode."),
            episode_shownotes: gr.update(value="", visible=False),
            transcription_output_df: gr.update(value=None),
            local_audio_file_path: None,
            transcribe_button: gr.update(interactive=False),
            selected_episode_index_state: None
        }

    try:
        episode = podcast_data.episodes[selected_episode_index]
        audio_url = episode.audio_url
        print(f"获取到剧集信息，标题: {episode.title}, 音频URL: {audio_url}")

        # 准备剧集信息显示
        episode_shownotes_content = ""
        
        # 准备shownotes内容
        if episode.shownotes:
            # 清理HTML标签并格式化shownotes
            import re
            # 简单的HTML标签清理
            clean_shownotes = re.sub(r'<[^>]+>', '', episode.shownotes)
            # 替换HTML实体
            clean_shownotes = clean_shownotes.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            # 清理多余空白
            clean_shownotes = re.sub(r'\s+', ' ', clean_shownotes).strip()
            
            episode_shownotes_content = f"### 📝 Episode Details\n\n**Title:** {episode.title or 'Untitled'}\n\n"
            if episode.published_date:
                episode_shownotes_content += f"**Published Date:** {episode.published_date.strftime('%Y-%m-%d')}\n\n"
            if episode.duration:
                episode_shownotes_content += f"**Duration:** {episode.duration}\n\n"
            
            episode_shownotes_content += f"**Episode Description:**\n\n{clean_shownotes}"
        elif episode.summary:
            # 如果没有shownotes，使用summary
            episode_shownotes_content = f"### 📝 Episode Details\n\n**Title:** {episode.title or 'Untitled'}\n\n"
            if episode.published_date:
                episode_shownotes_content += f"**Published Date:** {episode.published_date.strftime('%Y-%m-%d')}\n\n"
            if episode.duration:
                episode_shownotes_content += f"**Duration:** {episode.duration}\n\n"
            
            episode_shownotes_content += f"**Episode Summary:**\n\n{episode.summary}"
        else:
            # 最基本的信息
            episode_shownotes_content = f"### 📝 Episode Details\n\n**Title:** {episode.title or 'Untitled'}\n\n"
            if episode.published_date:
                episode_shownotes_content += f"**Published Date:** {episode.published_date.strftime('%Y-%m-%d')}\n\n"
            if episode.duration:
                episode_shownotes_content += f"**Duration:** {episode.duration}\n\n"

        if audio_url:
            # 更新状态消息
            print(f"正在下载音频: {audio_url}")
            
            # 下载音频文件
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
                }
                
                # 创建临时文件
                temp_dir = tempfile.gettempdir()
                # 使用会话ID和UUID生成唯一文件名，避免冲突
                unique_filename = f"podcast_audio_{session_id[:8]}_{uuid.uuid4().hex[:8]}"
                
                # 先发送一个HEAD请求获取内容类型和文件大小
                head_response = requests.head(audio_url, timeout=30, headers=headers)
                
                # 检查文件大小限制（例如限制为200MB）
                content_length = head_response.headers.get('Content-Length')
                if content_length:
                    file_size = int(content_length)
                    max_size = 200 * 1024 * 1024  # 200MB
                    if file_size > max_size:
                        return {
                            audio_player: gr.update(value=None),
                            current_audio_url_state: None,
                            status_message_area: gr.update(value=f"⚠️ 音频文件过大 ({file_size/1024/1024:.1f}MB)，超过限制 ({max_size/1024/1024}MB)"),
                            episode_shownotes: gr.update(value=episode_shownotes_content, visible=True),
                            transcription_output_df: gr.update(value=None),
                            local_audio_file_path: None,
                            transcribe_button: gr.update(interactive=False),
                            selected_episode_index_state: None
                        }
                
                # 根据内容类型确定文件扩展名
                content_type = head_response.headers.get('Content-Type', '').lower()
                if 'mp3' in content_type:
                    file_ext = '.mp3'
                elif 'mpeg' in content_type:
                    file_ext = '.mp3'
                elif 'mp4' in content_type or 'm4a' in content_type:
                    file_ext = '.mp4'
                elif 'wav' in content_type:
                    file_ext = '.wav'
                elif 'ogg' in content_type:
                    file_ext = '.ogg'
                else:
                    # 默认扩展名
                    file_ext = '.mp3'
                
                temp_filepath = os.path.join(temp_dir, unique_filename + file_ext)
                
                # 将文件路径添加到会话文件管理器
                file_manager.add_file(session_id, temp_filepath)
                
                # 保存到临时文件
                # 使用流式下载，避免一次性加载整个文件到内存
                with open(temp_filepath, 'wb') as f:
                    # 使用流式响应并设置较大的块大小提高效率
                    response = requests.get(audio_url, timeout=60, headers=headers, stream=True)
                    response.raise_for_status()
                    
                    # 从响应中获取文件大小（如果服务器提供）
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    chunk_size = 8192  # 8KB 的块大小
                    
                    # 分块下载并写入文件
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:  # 过滤掉保持连接活跃的空块
                            f.write(chunk)
                            downloaded += len(chunk)
                            # 可以在这里添加下载进度更新
                            if total_size > 0:
                                download_percentage = downloaded / total_size
                                print(f"下载进度: {download_percentage:.1%}")
                    
                print(f"音频已下载到临时文件: {temp_filepath}")
                
                return {
                    audio_player: gr.update(value=temp_filepath, label=f"Now Playing: {episode.title or 'Untitled'}"),
                    current_audio_url_state: audio_url,
                    status_message_area: gr.update(value=f"✅ Episode loaded: {episode.title or 'Untitled'}."),
                    episode_shownotes: gr.update(value=episode_shownotes_content, visible=True),
                    transcription_output_df: gr.update(value=None),
                    local_audio_file_path: temp_filepath,
                    transcribe_button: gr.update(interactive=True),
                    selected_episode_index_state: selected_episode_index
                }
            except requests.exceptions.RequestException as e:
                print(f"下载音频失败: {e}")
                traceback.print_exc()
                return {
                    audio_player: gr.update(value=None),
                    current_audio_url_state: None,
                    status_message_area: gr.update(value=f"❌ Error: Failed to download audio: {e}"),
                    episode_shownotes: gr.update(value=episode_shownotes_content, visible=True),
                    transcription_output_df: gr.update(value=None),
                    local_audio_file_path: None,
                    transcribe_button: gr.update(interactive=False),
                    selected_episode_index_state: None
                }
        else:
            print(f"剧集 '{episode.title}' 缺少有效的音频URL")
            return {
                audio_player: gr.update(value=None),
                current_audio_url_state: None,
                status_message_area: gr.update(value=f"❌ Error: Selected episode '{episode.title}' does not provide a valid audio URL."),
                episode_shownotes: gr.update(value=episode_shownotes_content, visible=True),
                transcription_output_df: gr.update(value=None),
                local_audio_file_path: None,
                transcribe_button: gr.update(interactive=False),
                selected_episode_index_state: None
            }
    except IndexError:
        print(f"无效的剧集索引: {selected_episode_index}")
        return {
            audio_player: gr.update(value=None),
            current_audio_url_state: None,
            status_message_area: gr.update(value="❌ Error: Invalid episode index selected."),
            episode_shownotes: gr.update(value="", visible=False),
            transcription_output_df: gr.update(value=None),
            local_audio_file_path: None,
            transcribe_button: gr.update(interactive=False),
            selected_episode_index_state: None
        }
    except Exception as e:
        print(f"加载音频时发生错误: {e}")
        traceback.print_exc()
        return {
            audio_player: gr.update(value=None),
            current_audio_url_state: None,
            status_message_area: gr.update(value=f"❌ Serious error occurred while loading audio: {e}"),
            episode_shownotes: gr.update(value="", visible=False),
            transcription_output_df: gr.update(value=None),
            local_audio_file_path: None,
            transcribe_button: gr.update(interactive=False),
            selected_episode_index_state: None
        }

def disable_buttons_before_transcription(local_audio_file_path: str):
    """在开始转录前禁用按钮"""
    print("禁用界面按钮以防止转录期间的交互")
    return {
        parse_button: gr.update(interactive=False),
        episode_dropdown: gr.update(interactive=False),
        transcribe_button: gr.update(interactive=False),
        status_message_area: gr.update(value="Starting transcription process, please wait...")
    }

def start_transcription(local_audio_file_path: str, podcast_data: PodcastChannel, selected_episode_index: int, progress=gr.Progress(track_tqdm=True)):
    """回调函数：开始转录当前加载的音频"""
    print(f"开始转录本地音频文件: {local_audio_file_path}, 选中剧集索引: {selected_episode_index}")
    
    # 检查系统资源
    resource_ok, resource_msg = check_system_resources()
    if not resource_ok:
        return {
            transcription_output_df: gr.update(value=None),
            status_message_area: gr.update(value=f"⚠️ {resource_msg}"),
            parse_button: gr.update(interactive=True),
            episode_dropdown: gr.update(interactive=True),
            transcribe_button: gr.update(interactive=True)
        }
    
    if not local_audio_file_path or not os.path.exists(local_audio_file_path):
        print("没有可用的本地音频文件")
        return {
            transcription_output_df: gr.update(value=None),
            status_message_area: gr.update(value="❌ Error: No valid audio file for transcription. Please select an episode first."),
            parse_button: gr.update(interactive=True),
            episode_dropdown: gr.update(interactive=True),
            transcribe_button: gr.update(interactive=True)
        }

    try:
        # 先更新状态消息并禁用按钮
        progress(0, desc="Initializing transcription process...")
        
        # 使用progress回调来更新进度
        progress(0.2, desc="Loading audio file...")
        
        # 从文件加载音频
        audio_segment = AudioSegment.from_file(local_audio_file_path)
        audio_duration = len(audio_segment) / 1000  # 转换为秒
        print(f"音频加载完成，时长: {audio_duration}秒")
        
        # 检查音频时长限制（例如限制为60分钟）
        max_duration = 60 * 60  # 60分钟
        if audio_duration > max_duration:
            return {
                transcription_output_df: gr.update(value=None),
                status_message_area: gr.update(value=f"⚠️ 音频时长过长 ({audio_duration/60:.1f}分钟)，超过限制 ({max_duration/60}分钟)"),
                parse_button: gr.update(interactive=True),
                episode_dropdown: gr.update(interactive=True),
                transcribe_button: gr.update(interactive=True)
            }
            
        progress(0.4, desc="Audio loaded, starting transcription (this may take a while)...")
        
        # 获取当前选中的剧集信息
        episode_info = None
        if podcast_data and podcast_data.episodes and selected_episode_index is not None:
            if 0 <= selected_episode_index < len(podcast_data.episodes):
                episode_info = podcast_data.episodes[selected_episode_index]
                print(f"获取到当前选中剧集信息: {episode_info.title if episode_info else '无'}")
        
        # 调用转录函数
        print("开始转录音频...")
        result: CombinedTranscriptionResult = transcribe_podcast_audio(audio_segment,
                                                                       podcast_info=podcast_data,
                                                                       episode_info=episode_info,
                                                                       segmentation_batch_size=32,  # 减少批次大小以节省内存
                                                                       parallel=True)
        print(f"转录完成，结果: {result is not None}, 段落数: {len(result.segments) if result and result.segments else 0}")
        progress(0.9, desc="Transcription completed, formatting results...")
        
        if result and result.segments:
            formatted_segments = []
            for seg in result.segments:
                time_str = f"{seg.start:.2f}s - {seg.end:.2f}s"
                formatted_segments.append([seg.speaker, seg.speaker_name, seg.text, time_str])
            
            progress(1.0, desc="Transcription results generated!")
            return {
                transcription_output_df: gr.update(value=formatted_segments),
                status_message_area: gr.update(value=f"✅ Transcription completed! {len(result.segments)} segments generated. {result.num_speakers} speakers detected."),
                parse_button: gr.update(interactive=True),
                episode_dropdown: gr.update(interactive=True),
                transcribe_button: gr.update(interactive=True)
            }
        elif result: # 有 result 但没有 segments
            progress(1.0, desc="Transcription completed, but no text segments")
            return {
                transcription_output_df: gr.update(value=None),
                status_message_area: gr.update(value="⚠️ Transcription completed, but no text segments were generated."),
                parse_button: gr.update(interactive=True),
                episode_dropdown: gr.update(interactive=True),
                transcribe_button: gr.update(interactive=True)
            }
        else: # result 为 None
            progress(1.0, desc="Transcription failed")
            return {
                transcription_output_df: gr.update(value=None),
                status_message_area: gr.update(value="❌ Transcription failed, no results obtained."),
                parse_button: gr.update(interactive=True),
                episode_dropdown: gr.update(interactive=True),
                transcribe_button: gr.update(interactive=True)
            }
    except Exception as e:
        print(f"转录过程中发生错误: {e}")
        traceback.print_exc()
        progress(1.0, desc="Transcription failed: processing error")
        return {
            transcription_output_df: gr.update(value=None),
            status_message_area: gr.update(value=f"❌ Serious error occurred during transcription: {e}"),
            parse_button: gr.update(interactive=True),
            episode_dropdown: gr.update(interactive=True),
            transcribe_button: gr.update(interactive=True)
        }

# --- Gradio 界面定义 ---
with gr.Blocks(title="Podcast Transcriber v2", css="""
.status-message-container {
    min-height: 50px;
    height: auto;
    max-height: none;
    overflow-y: visible;
    white-space: normal;
    word-wrap: break-word;
    margin-top: 10px;
    margin-bottom: 10px;
    border-radius: 6px;
    background-color: rgba(32, 36, 45, 0.03);
    border: 1px solid rgba(32, 36, 45, 0.1);
    color: #303030;
}
.episode-cover {
    max-width: 300px;
    max-height: 300px;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.resource-warning {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 6px;
    padding: 10px;
    margin: 10px 0;
    color: #856404;
}
""") as demo:
    gr.Markdown("# 🎙️ Podcast Transcriber")

    # 状态管理
    podcast_data_state = gr.State(None) # 存储解析后的 PodcastChannel 对象
    current_audio_url_state = gr.State(None) # 存储当前选中剧集的音频URL
    local_audio_file_path = gr.State(None) # 存储下载到本地的音频文件路径
    selected_episode_index_state = gr.State(None) # 存储当前选中的剧集索引

    with gr.Row():
        rss_url_input = gr.Textbox(
            label="Podcast RSS URL (Find RSS URL on https://castos.com/tools/find-podcast-rss-feed/)", 
            placeholder="e.g., https://your-podcast-feed.com/rss.xml",
            elem_id="rss-url-input"
        )
        parse_button = gr.Button("🔗 Parse RSS", elem_id="parse-rss-button")

    # 添加示例 RSS URL
    gr.Examples(
        examples=[["https://feeds.buzzsprout.com/2460059.rss"]],
        inputs=[rss_url_input],
        label="Example RSS URLs"
    )

    status_message_area = gr.Markdown(
        "", 
        elem_id="status-message",
        elem_classes="status-message-container",  # 添加自定义CSS类
        show_label=False
    )

    # 播客标题显示区域
    podcast_title_display = gr.Markdown(
        "",
        visible=False,
        elem_id="podcast-title-display"
    )

    episode_dropdown = gr.Dropdown(
        label="Select Episode", 
        choices=[], 
        interactive=False, # 初始时不可交互，解析成功后设为 True
        elem_id="episode-dropdown"
    )
    
    # 剧集信息显示区域
    with gr.Row():
        with gr.Column(scale=2):
            episode_shownotes = gr.Markdown(
                "",
                visible=False,
                elem_id="episode-shownotes"
            )
    
    audio_player = gr.Audio(
        label="Podcast Audio Player", 
        interactive=False, # 音频源由程序控制，用户不能直接修改
        elem_id="audio-player"
    )

    transcribe_button = gr.Button("🔊 Start Transcription", elem_id="transcribe-button", interactive=False)
    
    gr.Markdown("## 📝 Transcription Results")
    transcription_output_df = gr.DataFrame(
        headers=["Speaker ID", "Speaker Name", "Transcription Text", "Time Range"], 
        interactive=False,
        wrap=True, # 允许文本换行
        row_count=(10, "dynamic"), # 显示10行，可滚动
        col_count=(4, "fixed"),
        elem_id="transcription-output"
    )

    # --- 事件处理 ---
    parse_button.click(
        fn=parse_rss_feed,
        inputs=[rss_url_input],
        outputs=[
            status_message_area, 
            podcast_title_display,
            episode_dropdown, 
            podcast_data_state, 
            audio_player, 
            current_audio_url_state, 
            episode_shownotes,
            transcription_output_df,
            transcribe_button,
            selected_episode_index_state
        ]
    )

    episode_dropdown.change(
        fn=load_episode_audio,
        inputs=[episode_dropdown, podcast_data_state],
        outputs=[
            audio_player, 
            current_audio_url_state, 
            status_message_area, 
            episode_shownotes,
            transcription_output_df,
            local_audio_file_path,
            transcribe_button,
            selected_episode_index_state
        ]
    )

    # 首先禁用按钮，然后执行转录
    transcribe_button.click(
        fn=disable_buttons_before_transcription,
        inputs=[local_audio_file_path],
        outputs=[parse_button, episode_dropdown, transcribe_button, status_message_area]
    ).then(
        fn=start_transcription,
        inputs=[local_audio_file_path, podcast_data_state, selected_episode_index_state],
        outputs=[transcription_output_df, status_message_area, parse_button, episode_dropdown, transcribe_button]
    )

if __name__ == "__main__":
    try:
        # demo.launch(debug=True, share=True) # share=True 会生成一个公开链接
        demo.launch(debug=True)
    finally:
        # 确保在应用程序退出时清理临时文件
        cleanup_temp_files()

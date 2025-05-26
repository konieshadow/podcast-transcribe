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

# 用于存储应用程序使用的所有临时文件路径
temp_files = []

def cleanup_temp_files():
    """清理应用程序使用的临时文件"""
    global temp_files
    print(f"应用程序退出，清理 {len(temp_files)} 个临时文件...")
    
    for filepath in temp_files:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"已删除临时文件: {filepath}")
        except Exception as e:
            print(f"无法删除临时文件 {filepath}: {e}")
    
    # 清空列表
    temp_files = []

# 注册应用程序退出时的清理函数
atexit.register(cleanup_temp_files)

def parse_rss_feed(rss_url: str):
    """回调函数：解析 RSS Feed"""
    print(f"开始解析RSS: {rss_url}")
    
    if not rss_url:
        print("RSS地址为空")
        return {
            status_message_area: gr.update(value="错误：请输入 RSS 地址。"),
            episode_dropdown: gr.update(choices=[], value=None, interactive=False),
            podcast_data_state: None,
            audio_player: gr.update(value=None),
            current_audio_url_state: None,
            transcription_output_df: gr.update(value=None, headers=["说话人", "文本", "时间"]),
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
                label = f"{episode.title or '无标题'} (时长: {episode.duration or '未知'})"
                # 将 episode 对象直接作为值传递，或仅传递一个唯一标识符
                # 为了简单起见，我们使用索引作为唯一ID，因为我们需要从 podcast_data_state 中检索完整的 episode
                choices.append((label, i)) 
            
            return {
                status_message_area: gr.update(value=f"成功解析到 {len(podcast_data.episodes)} 个剧集。请选择一个剧集。"),
                episode_dropdown: gr.update(choices=choices, value=None, interactive=True),
                podcast_data_state: podcast_data,
                audio_player: gr.update(value=None),
                current_audio_url_state: None,
                transcription_output_df: gr.update(value=None),
                transcribe_button: gr.update(interactive=False),
                selected_episode_index_state: None
            }
        elif podcast_data: # 有 channel 信息但没有 episodes
            print("解析成功但未找到剧集")
            return {
                status_message_area: gr.update(value="解析成功，但未找到任何剧集。"),
                episode_dropdown: gr.update(choices=[], value=None, interactive=False),
                podcast_data_state: podcast_data, # 仍然存储，以防万一
                audio_player: gr.update(value=None),
                current_audio_url_state: None,
                transcription_output_df: gr.update(value=None),
                transcribe_button: gr.update(interactive=False),
                selected_episode_index_state: None
            }
        else:
            print(f"解析RSS失败: {rss_url}")
            return {
                status_message_area: gr.update(value=f"解析 RSS失败: {rss_url}。请检查URL或网络连接。"),
                episode_dropdown: gr.update(choices=[], value=None, interactive=False),
                podcast_data_state: None,
                audio_player: gr.update(value=None),
                current_audio_url_state: None,
                transcription_output_df: gr.update(value=None),
                transcribe_button: gr.update(interactive=False),
                selected_episode_index_state: None
            }
    except Exception as e:
        print(f"解析 RSS 时发生错误: {e}")
        traceback.print_exc()
        return {
            status_message_area: gr.update(value=f"解析 RSS 时发生严重错误: {e}"),
            episode_dropdown: gr.update(choices=[], value=None, interactive=False),
            podcast_data_state: None,
            audio_player: gr.update(value=None),
            current_audio_url_state: None,
            transcription_output_df: gr.update(value=None),
            transcribe_button: gr.update(interactive=False),
            selected_episode_index_state: None
        }

def load_episode_audio(selected_episode_index: int, podcast_data: PodcastChannel):
    """回调函数：当用户从下拉菜单选择一个剧集时加载音频"""
    global temp_files
    print(f"开始加载剧集音频，选择的索引: {selected_episode_index}")
    
    if selected_episode_index is None or podcast_data is None or not podcast_data.episodes:
        print("未选择剧集或无播客数据")
        return {
            audio_player: gr.update(value=None),
            current_audio_url_state: None,
            status_message_area: gr.update(value="请先解析 RSS 并选择一个剧集。"),
            transcription_output_df: gr.update(value=None),
            local_audio_file_path: None,
            transcribe_button: gr.update(interactive=False),
            selected_episode_index_state: None
        }

    try:
        episode = podcast_data.episodes[selected_episode_index]
        audio_url = episode.audio_url
        print(f"获取到剧集信息，标题: {episode.title}, 音频URL: {audio_url}")

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
                # 使用UUID生成唯一文件名，避免冲突
                unique_filename = f"podcast_audio_{uuid.uuid4().hex}"
                
                # 先发送一个HEAD请求获取内容类型
                head_response = requests.head(audio_url, timeout=30, headers=headers)
                
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
                
                # 将文件路径添加到全局临时文件列表
                temp_files.append(temp_filepath)
                
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
                    audio_player: gr.update(value=temp_filepath, label=f"当前播放: {episode.title or '无标题'}"),
                    current_audio_url_state: audio_url,
                    status_message_area: gr.update(value=f"已加载剧集: {episode.title or '无标题'}。"),
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
                    status_message_area: gr.update(value=f"错误：下载音频失败: {e}"),
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
                status_message_area: gr.update(value=f"错误：选中的剧集 '{episode.title}' 没有提供有效的音频URL。"),
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
            status_message_area: gr.update(value="错误：选择的剧集索引无效。"),
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
            status_message_area: gr.update(value=f"加载音频时发生严重错误: {e}"),
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
        status_message_area: gr.update(value="开始转录过程，请耐心等待...")
    }

def start_transcription(local_audio_file_path: str, podcast_data: PodcastChannel, selected_episode_index: int, progress=gr.Progress(track_tqdm=True)):
    """回调函数：开始转录当前加载的音频"""
    print(f"开始转录本地音频文件: {local_audio_file_path}, 选中剧集索引: {selected_episode_index}")
    
    if not local_audio_file_path or not os.path.exists(local_audio_file_path):
        print("没有可用的本地音频文件")
        return {
            transcription_output_df: gr.update(value=None),
            status_message_area: gr.update(value="错误：没有有效的音频文件用于转录。请先选择一个剧集。"),
            parse_button: gr.update(interactive=True),
            episode_dropdown: gr.update(interactive=True),
            transcribe_button: gr.update(interactive=True)
        }

    try:
        # 先更新状态消息并禁用按钮
        progress(0, desc="初始化转录过程...")
        
        # 使用progress回调来更新进度
        progress(0.2, desc="加载音频文件...")
        
        # 从文件加载音频
        audio_segment = AudioSegment.from_file(local_audio_file_path)
        print(f"音频加载完成，时长: {len(audio_segment)/1000}秒")
            
        progress(0.4, desc="音频加载完成，开始转录 (此过程可能需要较长时间)...")
        
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
                                                                       segmentation_batch_size=64,
                                                                       parallel=True)
        print(f"转录完成，结果: {result is not None}, 段落数: {len(result.segments) if result and result.segments else 0}")
        progress(0.9, desc="转录完成，正在格式化结果...")
        
        if result and result.segments:
            formatted_segments = []
            for seg in result.segments:
                time_str = f"{seg.start:.2f}s - {seg.end:.2f}s"
                formatted_segments.append([seg.speaker, seg.speaker_name, seg.text, time_str])
            
            progress(1.0, desc="转录结果已生成!")
            return {
                transcription_output_df: gr.update(value=formatted_segments),
                status_message_area: gr.update(value=f"转录完成！共 {len(result.segments)} 个片段。检测到 {result.num_speakers} 个说话人。"),
                parse_button: gr.update(interactive=True),
                episode_dropdown: gr.update(interactive=True),
                transcribe_button: gr.update(interactive=True)
            }
        elif result: # 有 result 但没有 segments
            progress(1.0, desc="转录完成，但无文本片段")
            return {
                transcription_output_df: gr.update(value=None),
                status_message_area: gr.update(value="转录完成，但未生成任何文本片段。"),
                parse_button: gr.update(interactive=True),
                episode_dropdown: gr.update(interactive=True),
                transcribe_button: gr.update(interactive=True)
            }
        else: # result 为 None
            progress(1.0, desc="转录失败")
            return {
                transcription_output_df: gr.update(value=None),
                status_message_area: gr.update(value="转录失败，未能获取结果。"),
                parse_button: gr.update(interactive=True),
                episode_dropdown: gr.update(interactive=True),
                transcribe_button: gr.update(interactive=True)
            }
    except Exception as e:
        print(f"转录过程中发生错误: {e}")
        traceback.print_exc()
        progress(1.0, desc="转录失败: 处理错误")
        return {
            transcription_output_df: gr.update(value=None),
            status_message_area: gr.update(value=f"转录过程中发生严重错误: {e}"),
            parse_button: gr.update(interactive=True),
            episode_dropdown: gr.update(interactive=True),
            transcribe_button: gr.update(interactive=True)
        }

# --- Gradio 界面定义 ---
with gr.Blocks(title="播客转录工具 v2", css="""
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
""") as demo:
    gr.Markdown("# 🎙️ 播客转录工具")

    # 状态管理
    podcast_data_state = gr.State(None) # 存储解析后的 PodcastChannel 对象
    current_audio_url_state = gr.State(None) # 存储当前选中剧集的音频URL
    local_audio_file_path = gr.State(None) # 存储下载到本地的音频文件路径
    selected_episode_index_state = gr.State(None) # 存储当前选中的剧集索引

    with gr.Row():
        rss_url_input = gr.Textbox(
            label="播客 RSS 地址", 
            placeholder="例如: https://your-podcast-feed.com/rss.xml",
            elem_id="rss-url-input"
        )
        parse_button = gr.Button("🔗 解析 RSS", elem_id="parse-rss-button")

    status_message_area = gr.Markdown(
        "", 
        elem_id="status-message",
        elem_classes="status-message-container",  # 添加自定义CSS类
        show_label=False
    )

    episode_dropdown = gr.Dropdown(
        label="选择剧集", 
        choices=[], 
        interactive=False, # 初始时不可交互，解析成功后设为 True
        elem_id="episode-dropdown"
    )
    
    audio_player = gr.Audio(
        label="播客音频播放器", 
        interactive=False, # 音频源由程序控制，用户不能直接修改
        elem_id="audio-player"
    )

    transcribe_button = gr.Button("🔊 开始转录", elem_id="transcribe-button", interactive=False)
    
    gr.Markdown("## 📝 转录结果")
    transcription_output_df = gr.DataFrame(
        headers=["说话人ID", "说话人名称", "转录文本", "起止时间"], 
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
            episode_dropdown, 
            podcast_data_state, 
            audio_player, 
            current_audio_url_state, 
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

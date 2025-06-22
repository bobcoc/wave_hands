import cv2
import yaml
import os
import sys
import time
from multiprocessing import Process
from datetime import datetime

# 读取配置文件
def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# 创建支持硬件解码的VideoCapture对象（复用detect_video.py中的函数）
def create_video_capture_with_hwdecode(url, config, name):
    """
    创建支持硬件解码的VideoCapture对象
    """
    video_decode_config = config.get('video_decode', {})
    use_hardware_decode = video_decode_config.get('use_hardware_decode', False)
    buffer_size = video_decode_config.get('buffer_size', 1)
    decode_backend = video_decode_config.get('decode_backend', 'ffmpeg')
    fallback_to_software = video_decode_config.get('fallback_to_software', True)
    rtsp_transport = video_decode_config.get('rtsp_transport', 'tcp')
    
    cap = None
    
    # 根据传输协议修改URL
    if rtsp_transport.lower() == 'tcp' and 'rtsp://' in url:
        # 对于TCP传输，可以在FFmpeg选项中设置
        modified_url = url
    else:
        modified_url = url
    
    if use_hardware_decode:
        try:
            print(f"[{name}] 尝试使用硬件解码器，传输协议: {rtsp_transport}")
            # 使用FFmpeg后端
            if decode_backend.lower() == 'ffmpeg':
                cap = cv2.VideoCapture(modified_url, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(modified_url)
            
            if cap.isOpened():
                # 设置缓冲区大小
                cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
                # 尝试设置硬件解码相关参数
                try:
                    # 设置解码器优化参数
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('H', '2', '6', '4'))
                    # 设置更多容错参数
                    cap.set(cv2.CAP_PROP_FPS, 25)  # 限制FPS减少缓冲压力
                    # 设置RTSP传输协议为TCP（更稳定但延迟更高）
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30000)  # 30秒超时
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 30000)  # 30秒读取超时
                except Exception as e:
                    print(f"[{name}] 硬件解码参数设置警告: {e}")
                
                print(f"[{name}] 硬件解码器初始化成功")
                return cap
            else:
                print(f"[{name}] 硬件解码器初始化失败")
                if cap:
                    cap.release()
                    cap = None
        except Exception as e:
            print(f"[{name}] 硬件解码器创建异常: {e}")
            if cap:
                cap.release()
                cap = None
    
    # 软件解码备用方案
    if cap is None and fallback_to_software:
        try:
            print(f"[{name}] 回退到软件解码器...")
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
                print(f"[{name}] 软件解码器初始化成功")
            else:
                print(f"[{name}] 软件解码器也初始化失败")
        except Exception as e:
            print(f"[{name}] 软件解码器创建异常: {e}")
    
    return cap

def record_camera_worker(stream_cfg, config, output_dir, record_duration_seconds=180):
    """
    单个摄像头录制工作进程
    
    Args:
        stream_cfg: 流配置信息
        config: 全局配置
        output_dir: 输出目录
        record_duration_seconds: 录制时长（秒），默认180秒（3分钟）
    """
    name = stream_cfg.get('name', 'noname')
    url = stream_cfg.get('url')
    
    print(f"[{name}] 录制进程启动，流地址: {url}")
    print(f"[{name}] 录制时长: {record_duration_seconds}秒")
    
    # 创建视频连接
    cap = create_video_capture_with_hwdecode(url, config, name)
    if not cap or not cap.isOpened():
        print(f"[{name}] 无法打开视频流: {url}")
        return False
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 如果fps为0或异常，设置默认值
    if fps <= 0 or fps > 60:
        fps = 25
        print(f"[{name}] 检测到异常帧率，设置为默认值: {fps} fps")
    
    print(f"[{name}] 视频属性: {width}x{height}, {fps} fps")
    
    # 创建输出文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"{name}_3min_{timestamp}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"[{name}] 无法创建输出视频文件: {output_path}")
        cap.release()
        return False
    
    print(f"[{name}] 开始录制到: {output_path}")
    
    # 录制参数
    start_time = time.time()
    frame_count = 0
    total_frames_expected = int(fps * record_duration_seconds)
    
    # 进度报告间隔（每30秒报告一次）
    last_progress_time = start_time
    progress_interval = 30
    
    try:
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # 检查录制时长
            if elapsed_time >= record_duration_seconds:
                print(f"[{name}] 录制完成，耗时: {elapsed_time:.1f}秒")
                break
            
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print(f"[{name}] 读取帧失败，尝试重连...")
                # 尝试重新连接
                cap.release()
                cap = create_video_capture_with_hwdecode(url, config, name)
                if not cap or not cap.isOpened():
                    print(f"[{name}] 重连失败，停止录制")
                    break
                continue
            
            # 写入帧
            out.write(frame)
            frame_count += 1
            
            # 进度报告
            if current_time - last_progress_time >= progress_interval:
                progress_percent = (elapsed_time / record_duration_seconds) * 100
                estimated_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                print(f"[{name}] 录制进度: {progress_percent:.1f}% ({elapsed_time:.0f}s/{record_duration_seconds}s), "
                      f"已录制: {frame_count} 帧, 实际帧率: {estimated_fps:.1f} fps")
                last_progress_time = current_time
    
    except KeyboardInterrupt:
        print(f"[{name}] 用户中断录制")
    except Exception as e:
        print(f"[{name}] 录制过程中发生错误: {e}")
    
    # 清理资源
    cap.release()
    out.release()
    
    # 检查文件是否成功创建
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        file_size_mb = file_size / (1024 * 1024)
        print(f"[{name}] 录制完成！")
        print(f"[{name}] 文件路径: {output_path}")
        print(f"[{name}] 文件大小: {file_size_mb:.1f} MB")
        print(f"[{name}] 总帧数: {frame_count}")
        return True
    else:
        print(f"[{name}] 录制失败，文件未生成")
        return False

def main():
    """主函数"""
    print("=== 多摄像头3分钟视频录制程序 ===")
    print("此程序将同时录制所有配置的摄像头3分钟视频")
    print("按 Ctrl+C 可提前停止录制")
    print()
    
    # 加载配置
    try:
        config = load_config()
    except Exception as e:
        print(f"读取配置文件失败: {e}")
        sys.exit(1)
    
    streams = config.get('streams', [])
    if not streams:
        print('配置文件未找到streams字段或为空')
        sys.exit(1)
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"test_videos_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 显示要录制的摄像头信息
    print(f"发现 {len(streams)} 个摄像头:")
    for i, stream in enumerate(streams, 1):
        name = stream.get('name', f'Camera_{i}')
        url = stream.get('url', 'Unknown')
        print(f"  {i}. {name}: {url}")
    print()
    
    # 确认开始录制
    try:
        input("按回车键开始录制 (Ctrl+C 取消)...")
    except KeyboardInterrupt:
        print("\n用户取消录制")
        sys.exit(0)
    
    print("\n开始录制...")
    
    # 创建并启动录制进程
    processes = []
    for stream_cfg in streams:
        p = Process(target=record_camera_worker, args=(stream_cfg, config, output_dir, 180))
        p.start()
        processes.append(p)
    
    # 等待所有进程完成
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n用户中断，正在停止所有录制进程...")
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                    p.join()
    
    print("\n=== 录制任务完成 ===")
    
    # 检查输出结果
    if os.path.exists(output_dir):
        video_files = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
        if video_files:
            print(f"成功录制 {len(video_files)} 个视频文件:")
            total_size = 0
            for video_file in video_files:
                file_path = os.path.join(output_dir, video_file)
                file_size = os.path.getsize(file_path)
                file_size_mb = file_size / (1024 * 1024)
                total_size += file_size_mb
                print(f"  - {video_file} ({file_size_mb:.1f} MB)")
            print(f"总大小: {total_size:.1f} MB")
            print(f"输出目录: {os.path.abspath(output_dir)}")
        else:
            print("未发现录制的视频文件")
    
    print("\n这些视频可用于后续的挥手检测问题排查")

if __name__ == '__main__':
    main() 
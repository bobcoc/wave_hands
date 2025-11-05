import cv2
import yaml
import os
import sys
import time
import glob
from hand_detector import HandDetector

def load_config(config_path='config.yaml'):
    """读取配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_video_files(directory):
    """递归搜索目录中的视频文件"""
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm', '*.m4v']
    video_files = []
    
    for ext in video_extensions:
        # 递归搜索所有子目录
        pattern = os.path.join(directory, '**', ext)
        video_files.extend(glob.glob(pattern, recursive=True))
    
    return sorted(video_files)

def detect_local_video(video_path, config):
    """
    检测本地视频文件中的手掌
    """
    name = 'local_video'
    weights = config.get('weights', 'weight/best.pt')
    confidence = float(config.get('confidence', 0.2))
    device = config.get('device', 'cpu')
    alarm_dir = config.get('alarm_dir', 'alarms')
    alarm_duration = int(config.get('alarm_duration', 3))
    alarm_video_overlay_level = int(config.get('alarm_video_overlay_level', 2))
    font_scale = float(config.get('font_scale', 0.4))
    font_thickness = int(config.get('font_thickness', 1))
    margin_ratio = float(config.get('margin_ratio', 0.02))  # 添加边距比例配置

    if not os.path.exists(alarm_dir):
        os.makedirs(alarm_dir)

    # 检查视频文件是否存在
    if not os.path.isfile(video_path):
        print(f"错误：视频文件不存在: {video_path}")
        return

    print(f"[{name}] 开始检测本地视频: {video_path}")

    # 初始化检测器
    detector = HandDetector(
        detector='mediapipe',  # 使用YOLO检测手掌位置，MediaPipe提取关键点
        weights=weights,
        confidence=confidence,
        device=device,
        font_scale=font_scale,
        font_thickness=font_thickness,
        margin_ratio=margin_ratio  # 传入边距比例
    )

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[{name}] 无法打开视频文件: {video_path}")
        return

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps > 0:
        alarm_buf_len = int(alarm_duration * fps)
    else:
        alarm_buf_len = 75  # 默认3秒*25帧

    print(f"[{name}] 视频信息: fps={fps}, 分辨率={width}x{height}, 总帧数={total_frames}")

    # 状态变量
    state = 'idle'  # idle, active
    global_frame_counter = 0
    active_buffer = []
    palm_frame_count = 0
    active_frame_counter = 0
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')

    print(f"[{name}] 开始逐帧检测...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[{name}] 视频读取完毕，总处理帧数: {global_frame_counter}")
            break

        global_frame_counter += 1

        if state == 'idle':
            # idle模式：检测手掌
            output, hands_info = detector.process_frame(frame)
            palm_found = len(hands_info) > 0
            
            print(f"[{name}] [idle] 帧 {global_frame_counter}/{total_frames} - 检测到手掌: {palm_found}")

            if palm_found:
                print(f"[{name}] 状态切换: idle -> active (检测到palm)")
                state = 'active'
                active_buffer = [frame.copy()]
                palm_frame_count = 1
                active_frame_counter = 1
                continue

        elif state == 'active':
            # active模式：继续检测并记录
            output, hands_info = detector.process_frame(frame)
            palm_in_this_frame = len(hands_info) > 0
            
            if palm_in_this_frame:
                palm_frame_count += 1

            active_buffer.append(frame.copy())
            active_frame_counter += 1

            print(f"[{name}] [active] 帧 {global_frame_counter}/{total_frames} - palm帧数: {palm_frame_count}/{active_frame_counter}")

            # 检查是否满足报警条件
            if active_frame_counter >= alarm_buf_len:
                if palm_frame_count >= 30:
                    # 输出报警视频片段
                    ts = time.strftime('%Y%m%d_%H%M%S')
                    alarm_path = os.path.join(alarm_dir, f'{name}_{ts}.mp4')
                    alarm_writer = cv2.VideoWriter(alarm_path, fourcc, fps if fps > 0 else 25, (width, height))
                    
                    for idx, f in enumerate(active_buffer):
                        if alarm_video_overlay_level > 0:
                            # 使用detector处理每一帧
                            processed_frame, _ = detector.process_frame(f)
                            alarm_writer.write(processed_frame)
                        else:
                            alarm_writer.write(f)
                    
                    alarm_writer.release()
                    print(f"[{name}] !!! 触发报警：输出片段: {alarm_path}")
                    print(f"[{name}] 报警片段信息: {len(active_buffer)}帧, palm帧数: {palm_frame_count}")
                else:
                    print(f"[{name}] 3秒内palm帧数{palm_frame_count}<10，丢弃片段，回到idle")
                
                # 重置状态
                state = 'idle'
                active_buffer = []
                palm_frame_count = 0
                active_frame_counter = 0
                continue

    # 清理资源
    cap.release()
    print(f"[{name}] 本地视频检测完成")

def main():
    config = load_config()
    
    # 指定要处理的视频目录
    video_directory = "/Users/liushuming/projects/cc/vv"  
    if not os.path.exists(video_directory):
        print(f'错误：视频目录不存在: {video_directory}')
        sys.exit(1)
    
    # 获取所有视频文件
    video_files = get_video_files(video_directory)
    
    if not video_files:
        print(f'在目录 {video_directory} 中未找到视频文件')
        sys.exit(1)
    
    print(f'找到 {len(video_files)} 个视频文件:')
    for i, video_file in enumerate(video_files, 1):
        print(f'  {i}. {video_file}')
    
    print(f'\n开始处理视频文件...')
    
    # 逐一处理每个视频文件
    for i, video_file in enumerate(video_files, 1):
        print(f'\n[{i}/{len(video_files)}] 处理视频: {video_file}')
        try:
            detect_local_video(video_file, config)
        except Exception as e:
            print(f'处理视频 {video_file} 时出错: {e}')
            continue
    
    print(f'\n所有视频处理完成！')

if __name__ == '__main__':
    main() 
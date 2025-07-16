import cv2
import yaml
from ultralytics import YOLO
import os
import sys
import time
import numpy as np
from hand_detection_core import classify_hand_pose, draw_hand_overlay

# 读取配置文件
def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def detect_local_video(video_path, config):
    """
    检测本地视频文件中的手掌
    """
    name = 'local_video'
    weights = config.get('weights', 'weight/best.pt')
    confidence = float(config.get('confidence', 0.5))
    device = config.get('device', 'cpu')
    alarm_dir = config.get('alarm_dir', 'alarms')
    alarm_duration = int(config.get('alarm_duration', 3))
    alarm_video_overlay_level = int(config.get('alarm_video_overlay_level', 0))  # 0=不叠加，1=画palm框，2=画palm框和landmarks

    if not os.path.exists(alarm_dir):
        os.makedirs(alarm_dir)

    # 检查视频文件是否存在
    if not os.path.isfile(video_path):
        print(f"错误：视频文件不存在: {video_path}")
        return

    print(f"[{name}] 开始检测本地视频: {video_path}")

    # 加载模型
    model = YOLO(weights, task='detect')
    model.to(device)

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

    # MediaPipe手部骨架连线
    mediapipe_connections = [
        (0,1),(1,2),(2,3),(3,4),    # 大拇指
        (0,5),(5,6),(6,7),(7,8),    # 食指
        (0,9),(9,10),(10,11),(11,12), # 中指
        (0,13),(13,14),(14,15),(15,16), # 无名指
        (0,17),(17,18),(18,19),(19,20)  # 小指
    ]

    # 状态变量
    state = 'idle'  # idle, active
    global_frame_counter = 0
    active_buffer = []
    palm_frame_count = 0
    active_frame_counter = 0
    last_palm_roi = None
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
            results = model(frame, conf=confidence, verbose=False)[0]
            frame_to_show = draw_hand_overlay(frame, results, alarm_video_overlay_level, mediapipe_connections)
            
            palm_found = False
            palm_roi = None
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:  # palm class
                    palm_found = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    palm_roi = (x1, y1, x2, y2)
                    break

            print(f"[{name}] [idle] 帧 {global_frame_counter}/{total_frames} - 检测到手掌: {palm_found}")

            if palm_found:
                last_palm_roi = palm_roi
                print(f"[{name}] 状态切换: idle -> active (检测到palm)")
                state = 'active'
                active_buffer = [frame.copy()]
                palm_frame_count = 1
                active_frame_counter = 1
                continue

        elif state == 'active':
            # active模式：继续检测并记录
            if last_palm_roi is None:
                print(f"[{name}] 警告：active状态下last_palm_roi无效，回退idle")
                state = 'idle'
                continue

            # 在ROI区域内检测
            x1, y1, x2, y2 = last_palm_roi
            roi_x1 = max(0, x1-300)
            roi_y1 = max(0, y1-300)
            roi_x2 = min(width, x2+300)
            roi_y2 = min(height, y2+300)
            
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            results = model(roi, conf=confidence, verbose=False)[0]
            
            palm_in_this_frame = False
            new_palm_roi = None
            for i, box in enumerate(results.boxes):
                cls_id = int(box.cls[0])
                if cls_id == 0:
                    palm_in_this_frame = True
                    rx1, ry1, rx2, ry2 = map(int, box.xyxy[0])
                    x1g, y1g, x2g, y2g = rx1+roi_x1, ry1+roi_y1, rx2+roi_x1, ry2+roi_y1
                    new_palm_roi = (x1g, y1g, x2g, y2g)
                    break

            frame_to_show = draw_hand_overlay(frame, results, alarm_video_overlay_level, mediapipe_connections)
            
            if palm_in_this_frame and new_palm_roi is not None:
                last_palm_roi = new_palm_roi
                palm_frame_count += 1

            active_buffer.append(frame.copy())
            active_frame_counter += 1

            print(f"[{name}] [active] 帧 {global_frame_counter}/{total_frames} - palm帧数: {palm_frame_count}/{active_frame_counter}")

            # 检查是否满足报警条件
            if palm_frame_count >= 30 or active_frame_counter >= alarm_buf_len:
                if palm_frame_count >= 30:
                    # 输出报警视频片段
                    ts = time.strftime('%Y%m%d_%H%M%S')
                    alarm_path = os.path.join(alarm_dir, f'{name}_{ts}.mp4')
                    alarm_writer = cv2.VideoWriter(alarm_path, fourcc, fps if fps > 0 else 25, (width, height))
                    
                    for idx, f in enumerate(active_buffer):
                        if alarm_video_overlay_level == 0:
                            alarm_writer.write(f)
                        else:
                            results = model(f, conf=confidence, verbose=False)[0]
                            overlayed = draw_hand_overlay(f, results, alarm_video_overlay_level, mediapipe_connections)
                            alarm_writer.write(overlayed)
                    
                    alarm_writer.release()
                    print(f"[{name}] !!! 触发报警：输出片段: {alarm_path}")
                    print(f"[{name}] 报警片段信息: {len(active_buffer)}帧, palm帧数: {palm_frame_count}")
                else:
                    print(f"[{name}] 3秒内palm帧数{palm_frame_count}<30，丢弃片段，回到idle")
                
                # 重置状态
                state = 'idle'
                active_buffer = []
                palm_frame_count = 0
                active_frame_counter = 0
                last_palm_roi = None
                continue

    # 清理资源
    cap.release()
    print(f"[{name}] 本地视频检测完成")

def main():
    config = load_config()
    video_file = config.get('video_file', None)
    
    if not video_file or not str(video_file).strip():
        print('错误：配置文件中未找到video_file字段或为空')
        print('请在config.yaml中添加video_file字段，例如:')
        print('video_file: "path/to/your/video.mp4"')
        sys.exit(1)
    
    detect_local_video(video_file, config)

if __name__ == '__main__':
    main() 
import cv2
import os
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from collections import deque, defaultdict
import math
import yaml
from deep_sort_realtime.deepsort_tracker import DeepSort

def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def detect_dimension_changes(width_history, height_history, change_threshold=30):
    """检测宽度或高度的剧烈变化次数"""
    if len(width_history) < 2 or len(height_history) < 2:
        return 0, []
    
    changes = []
    change_count = 0
    
    # 检测宽度变化
    for i in range(1, len(width_history)):
        prev_width = width_history[i-1]
        curr_width = width_history[i]
        width_change = abs(curr_width - prev_width)
        
        if width_change >= change_threshold:
            change_count += 1
            changes.append(('width_change', i, prev_width, curr_width, width_change))
    
    # 检测高度变化
    for i in range(1, len(height_history)):
        prev_height = height_history[i-1]
        curr_height = height_history[i]
        height_change = abs(curr_height - prev_height)
        
        if height_change >= change_threshold:
            change_count += 1
            changes.append(('height_change', i, prev_height, curr_height, height_change))
    
    return change_count, changes

def main():
    # 读取配置
    config = load_config()
    video_path = config.get('input_video', 'input.mp4')
    output_dir = config.get('alarm_dir', 'alarms')
    os.makedirs(output_dir, exist_ok=True)
    alarm_duration = int(config.get('alarm_duration', 3))
    wave_change_threshold = int(config.get('wave_change_threshold', 3))
    confidence = float(config.get('confidence', 0.7))
    device = config.get('device', 'cpu')
    teacher_height_threshold = int(config.get('teacher_height_threshold', 600))
    name = os.path.splitext(os.path.basename(video_path))[0]

    # 初始化YOLO
    model = YOLO(config.get('weights', 'yolov8n.pt'))
    model.to(device)

    # 初始化DeepSORT
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, embedder="mobilenet", half=True, bgr=True)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 确保使用75帧作为滑动窗口（3秒×25fps）
    window_len = 75

    print(f"=== DeepSORT Wave Detection Analysis ===")
    print(f"Video file: {video_path}")
    print(f"Video properties: {width}x{height}, {fps:.1f}fps, {total_frames} frames")
    print(f"Sliding window: {window_len} frames (3 seconds), crossing threshold: {wave_change_threshold}")
    print(f"Crossing detection: <0.3 ↔ >0.5 (bidirectional)")
    print(f"Analysis started... (press 'q' to quit)")

    # 维护每个track_id的宽度和高度历史
    width_history = defaultdict(lambda: deque(maxlen=window_len))
    height_history = defaultdict(lambda: deque(maxlen=window_len))
    
    # 教师过滤阈值（从配置文件读取，默认400像素）
    TEACHER_HEIGHT_THRESHOLD = teacher_height_threshold
    frame_buffer = deque(maxlen=window_len)
    alarm_active = False

    # Create aspect ratio data log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    aspect_log_path = os.path.join(output_dir, f"{name}_aspect_ratios_{timestamp}.txt")
    aspect_log_file = open(aspect_log_path, 'w', encoding='utf-8')
    aspect_log_file.write("# Dimension data log\n")
    aspect_log_file.write("# Format: frame_id,track_id,width,height,aspect_ratio,bbox(x1,y1,x2,y2)\n")
    
    # Create processed video output
    output_video_dir = "bbox_wave_analysis"
    os.makedirs(output_video_dir, exist_ok=True)
    output_video_path = os.path.join(output_video_dir, f"{name}_bbox_wave_{timestamp}.mp4")
    
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    output_video_writer = cv2.VideoWriter(output_video_path, fourcc, fps if fps > 0 else 25, (width, height))
    alarm_writer = None
    
    print(f"Aspect ratio data will be saved to: {aspect_log_path}")
    print(f"Processed video will be saved to: {output_video_path}")

    frame_idx = 0
    no_person_count = 0
    NO_PERSON_WARN_THRESHOLD = 50  # Warn when no person detected for 50 consecutive frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame_disp = frame.copy()
        frame_buffer.append(frame.copy())

        # YOLO detection
        results = model(frame, conf=0.3, verbose=False)[0]  # Lower confidence threshold
        boxes = results.boxes

        dets = []
        print(f"Frame {frame_idx} detection results:")
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                print(f"  Detected class: {cls_id}, confidence: {conf_score:.2f}, coords: ({x1},{y1},{x2},{y2})")
                # Only track persons (COCO model person class)
                if cls_id == 0:
                    dets.append(([x1, y1, x2 - x1, y2 - y1], conf_score, 'person'))
        else:
            print("  No detection results")
        print(f"Frame {frame_idx}: Detected {len(dets)} persons")
        if len(dets) == 0:
            no_person_count += 1
            if no_person_count == NO_PERSON_WARN_THRESHOLD:
                print(f"⚠️ Warning: No person detected for {NO_PERSON_WARN_THRESHOLD} consecutive frames, please check video content or model parameters!")
        else:
            no_person_count = 0

        # DeepSORT tracking
        tracks = tracker.update_tracks(dets, frame=frame)
        waving_ids = set()
        print(f"  DeepSORT tracked {len(tracks)} trajectories")
        for track in tracks:
            print(f"    Track {track.track_id}: confirmed={track.is_confirmed()}")
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            width_box = x2 - x1
            height_box = y2 - y1
            if height_box <= 0:
                continue
            
            # 过滤：只跟踪教师（高度>400像素）
            if height_box < TEACHER_HEIGHT_THRESHOLD:
                print(f"    ID{track_id}: Student filtered out (height={height_box} < {TEACHER_HEIGHT_THRESHOLD})")
                continue
            
            # 记录宽度和高度历史
            width_history[track_id].append(width_box)
            height_history[track_id].append(height_box)
            
            # 计算宽高比用于记录
            aspect = width_box / height_box
            
            # Record dimension data to file
            aspect_log_file.write(f"{frame_idx},{track_id},{width_box},{height_box},{aspect:.6f},{x1},{y1},{x2},{y2}\n")
            aspect_log_file.flush()  # Real-time writing
            
            print(f"    ID{track_id}: Teacher detected - W:{width_box}, H:{height_box}, ratio:{aspect:.3f}, history_length:{len(width_history[track_id])}")

            # 新的尺寸变化检测逻辑
            width_hist = list(width_history[track_id])
            height_hist = list(height_history[track_id])
            
            # Output complete 75-frame data analysis
            if len(width_hist) >= 20:  # Detailed analysis when enough data available
                print(f"    ID{track_id}: Complete 75-frame window analysis:")
                print(f"      Total frames: {len(width_hist)}")
                print(f"      Width range: {min(width_hist)}-{max(width_hist)} pixels")
                print(f"      Height range: {min(height_hist)}-{max(height_hist)} pixels")
                
                # Display recent history data (last 10 frames)
                print(f"      Recent 10 frames:")
                recent_width = width_hist[-10:] if len(width_hist) > 10 else width_hist
                recent_height = height_hist[-10:] if len(height_hist) > 10 else height_hist
                for i in range(len(recent_width)):
                    frame_num = frame_idx - len(recent_width) + i + 1
                    print(f"        Frame{frame_num}: W={recent_width[i]}, H={recent_height[i]}")
            else:
                # Simple display when data insufficient
                print(f"    ID{track_id}: History length insufficient, need at least 20 frames for complete analysis")
                print(f"    ID{track_id}: Current data - W:{width_hist[-1] if width_hist else 'N/A'}, H:{height_hist[-1] if height_hist else 'N/A'}")
            
            # Detect dimension changes (宽度或高度变化超过30像素)
            change_count, changes = detect_dimension_changes(width_hist, height_hist, change_threshold=30)
            
            print(f"    ID{track_id}: Dimension changes in 75-frame window={change_count}, threshold={wave_change_threshold}")
            
            # Display change details
            if changes:
                print(f"    ID{track_id}: Change details:")
                for change_type, frame_pos, prev_val, curr_val, change_magnitude in changes:
                    actual_frame = frame_idx - len(width_hist) + frame_pos
                    print(f"      Frame{actual_frame}: {change_type} {prev_val}→{curr_val} (change={change_magnitude}px)")
            
            # Calculate statistics
            if len(width_hist) > 0:
                width_range = max(width_hist) - min(width_hist)
                height_range = max(height_hist) - min(height_hist)
                print(f"    ID{track_id}: Variation - Width:{width_range}px, Height:{height_range}px")
            
            # Determine if waving (宽度或高度有剧烈变化)
            if change_count >= wave_change_threshold:
                waving_ids.add(track_id)
                color = (0, 255, 0)
                status = "WAVING"
                print(f"    *** ID{track_id} DETECTED AS WAVING! ***")
            else:
                color = (255, 0, 0)
                status = "NORMAL"
            
            # Draw bounding box
            cv2.rectangle(frame_disp, (x1, y1), (x2, y2), color, 2)
            
            # Display detailed information on the video with white background and black text
            # Calculate text background size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Prepare labels
            labels = [
                f"ID{track_id}: {status} (Teacher)",
                f"W:{width_box} H:{height_box}",
                f"Ratio:{aspect:.3f} Changes:{change_count}",
                f"History:{len(width_history[track_id])}"
            ]
            
            # Calculate maximum text width for background
            max_width = 0
            for label in labels:
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                max_width = max(max_width, text_width)
            
            # Draw white background for text
            bg_height = len(labels) * 25 + 10
            cv2.rectangle(frame_disp, (x1-5, y1-bg_height), (x1+max_width+10, y1+5), (255, 255, 255), -1)
            cv2.rectangle(frame_disp, (x1-5, y1-bg_height), (x1+max_width+10, y1+5), (0, 0, 0), 1)
            
            # Draw text with black color
            for i, label in enumerate(labels):
                y_pos = y1 - bg_height + 20 + i * 20
                cv2.putText(frame_disp, label, (x1, y_pos), font, font_scale, (0, 0, 0), thickness)

        # Add global information overlay on the video with white background and black text
        # Global statistics
        info_lines = [
            f"Frame: {frame_idx}/{total_frames}",
            f"Teachers detected: {len([t for t in tracks if t.is_confirmed()])}",
            f"Waving detected: {len(waving_ids)}",
            f"Filter: Height>400px, Change>30px"
        ]
        
        # Calculate background size for global info
        global_font = cv2.FONT_HERSHEY_SIMPLEX
        global_font_scale = 0.6
        global_thickness = 2
        
        max_global_width = 0
        for info_line in info_lines:
            (text_width, text_height), _ = cv2.getTextSize(info_line, global_font, global_font_scale, global_thickness)
            max_global_width = max(max_global_width, text_width)
        
        # Draw white background for global info
        global_bg_height = len(info_lines) * 25 + 20
        cv2.rectangle(frame_disp, (10, 10), (max_global_width + 30, global_bg_height), (255, 255, 255), -1)
        cv2.rectangle(frame_disp, (10, 10), (max_global_width + 30, global_bg_height), (0, 0, 0), 2)
        
        # Draw global info text with black color
        for i, info_line in enumerate(info_lines):
            cv2.putText(frame_disp, info_line, (15, 35 + i*20), global_font, global_font_scale, (0, 0, 0), global_thickness)

        # Save processed video frame
        output_video_writer.write(frame_disp)

        # Detected waving, save alarm segment
        if waving_ids and not alarm_active:
            alarm_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            alarm_path = os.path.join(output_dir, f"{name}_{alarm_timestamp}.mp4")
            alarm_writer = cv2.VideoWriter(alarm_path, fourcc, fps if fps > 0 else 25, (width, height))
            for f in frame_buffer:
                alarm_writer.write(f)
            alarm_writer.release()
            print(f"!!! WAVING DETECTED, alarm segment saved: {alarm_path}")
            alarm_active = True
        if not waving_ids:
            alarm_active = False

        # Progress output every 200 frames
        if frame_idx % 200 == 0:
            print(f"Progress: {frame_idx}/{total_frames} frames ({(frame_idx/total_frames)*100:.1f}%)")

        # Display
        cv2.imshow('Wave Detection (DeepSORT)', frame_disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    cap.release()
    output_video_writer.release()
    aspect_log_file.close()
    cv2.destroyAllWindows()
    
    print(f"\n=== Processing Complete ===")
    print(f"Aspect ratio data saved: {aspect_log_path}")
    print(f"Processed video saved: {output_video_path}")
    
    # Verify output files
    if os.path.exists(aspect_log_path):
        file_size = os.path.getsize(aspect_log_path) / 1024  # KB
        print(f"Aspect ratio data file size: {file_size:.1f} KB")
    
    if os.path.exists(output_video_path):
        file_size = os.path.getsize(output_video_path) / (1024*1024)  # MB
        print(f"Processed video file size: {file_size:.1f} MB")
if __name__ == '__main__':
    main()
import cv2
import os
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from collections import deque, defaultdict
import math

class SimpleWaveDetector:
    """基于宽高比的简化挥手检测器"""
    
    def __init__(self, wave_aspect_ratio_min=0.7, min_wave_duration=2):
        self.wave_aspect_ratio_min = wave_aspect_ratio_min
        self.min_wave_duration = min_wave_duration
        self.wave_counters = defaultdict(int)
        
    def detect_wave(self, track_id, bbox):
        """检测挥手：基于宽高比"""
        x1, y1, x2, y2, conf = bbox
        width = x2 - x1
        height = y2 - y1
        
        if height <= 0:
            return False, 0
        
        aspect_ratio = width / height
        
        # 判断当前帧是否为挥手状态
        is_current_wave = aspect_ratio >= self.wave_aspect_ratio_min
        
        if is_current_wave:
            self.wave_counters[track_id] += 1
        else:
            self.wave_counters[track_id] = 0
        
        # 持续挥手才算真正的挥手
        is_waving = self.wave_counters[track_id] >= self.min_wave_duration
        
        return is_waving, aspect_ratio

def analyze_video_with_fixed_encoding(video_path, show_video=True, save_output=True):
    """修复了编码问题的视频分析"""
    print(f"=== 修复版挥手检测 ===")
    print(f"视频文件: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在")
        return False
    
    # 加载YOLO人物检测模型
    model = YOLO('yolov8n.pt')
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频")
        return False
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频属性: {width}x{height}, {fps:.1f}fps, {total_frames}帧")
    
    # 修复帧率问题
    if fps <= 0 or fps > 120:
        fps = 25.0
        print(f"修复帧率为: {fps}")
    
    # 初始化检测器
    wave_detector = SimpleWaveDetector(wave_aspect_ratio_min=0.7, min_wave_duration=2)
    
    # 准备输出视频 - 使用多种编码器尝试
    output_writer = None
    output_path = None
    
    if save_output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 创建输出目录
        output_dir = "fixed_wave_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        # 尝试不同的编码器和格式
        encoders_to_try = [
            ('avi', 'XVID'),  # 最兼容的组合
            ('mp4', 'mp4v'),  # 标准MP4
            ('mov', 'mp4v'),  # MOV格式
        ]
        
        for ext, codec in encoders_to_try:
            output_path = os.path.join(output_dir, f"{video_name}_fixed_wave_{timestamp}.{ext}")
            fourcc = cv2.VideoWriter.fourcc(*codec)
            output_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if output_writer.isOpened():
                print(f"✅ 输出视频: {output_path} (编码器: {codec})")
                break
            else:
                print(f"❌ 编码器 {codec} 失败，尝试下一个...")
                output_writer = None
        
        if output_writer is None:
            print("⚠️ 所有编码器都失败，将不保存视频")
    
    # 简单的人物跟踪 - 使用位置距离
    person_tracks = {}
    next_id = 0
    max_distance = 100
    
    # 统计信息
    stats = {
        'total_frames': 0,
        'frames_with_persons': 0,
        'frames_with_waves': 0,
        'high_aspect_ratio_count': 0
    }
    
    frame_count = 0
    
    print(f"\n开始分析... (按 'q' 退出)")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            stats['total_frames'] = frame_count
            
            # 使用YOLO检测人物
            results = model(frame, conf=0.3, verbose=False)[0]
            boxes = results.boxes
            
            # 提取人物检测结果
            current_detections = []
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if cls_id == 0:  # 人类
                        conf_score = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        current_detections.append((x1, y1, x2, y2, conf_score, center_x, center_y))
            
            if current_detections:
                stats['frames_with_persons'] += 1
            
            # 简单的跟踪：匹配最近的检测结果
            matched_tracks = {}
            unmatched_detections = current_detections.copy()
            
            # 为现有轨迹找匹配
            for track_id, last_center in person_tracks.items():
                best_match = None
                best_distance = float('inf')
                
                for i, detection in enumerate(unmatched_detections):
                    x1, y1, x2, y2, conf, center_x, center_y = detection
                    distance = math.sqrt((center_x - last_center[0])**2 + (center_y - last_center[1])**2)
                    
                    if distance < max_distance and distance < best_distance:
                        best_distance = distance
                        best_match = i
                
                if best_match is not None:
                    detection = unmatched_detections.pop(best_match)
                    matched_tracks[track_id] = detection
            
            # 为未匹配的检测创建新轨迹
            for detection in unmatched_detections:
                matched_tracks[next_id] = detection
                next_id += 1
            
            # 更新轨迹
            person_tracks = {}
            current_frame_has_wave = False
            
            for track_id, detection in matched_tracks.items():
                x1, y1, x2, y2, conf, center_x, center_y = detection
                person_tracks[track_id] = (center_x, center_y)
                
                # 检测挥手
                bbox = (x1, y1, x2, y2, conf)
                is_waving, aspect_ratio = wave_detector.detect_wave(track_id, bbox)
                
                # 统计高宽高比
                if aspect_ratio >= 0.7:
                    stats['high_aspect_ratio_count'] += 1
                
                # 绘制结果
                if is_waving:
                    color = (0, 255, 0)  # 绿色 - 挥手
                    label = f"ID{track_id}: WAVING ({aspect_ratio:.2f})"
                    current_frame_has_wave = True
                elif aspect_ratio >= 0.7:
                    color = (0, 255, 255)  # 黄色 - 可能挥手
                    label = f"ID{track_id}: MAYBE ({aspect_ratio:.2f})"
                else:
                    color = (255, 0, 0)  # 蓝色 - 正常
                    label = f"ID{track_id}: Normal ({aspect_ratio:.2f})"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if current_frame_has_wave:
                stats['frames_with_waves'] += 1
            
            # 显示统计信息
            info_text = f"Frame: {frame_count}/{total_frames} | Persons: {len(matched_tracks)} | Waves: {stats['frames_with_waves']}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            threshold_text = f"Threshold: Aspect Ratio >= 0.7, Duration >= 2 frames"
            cv2.putText(frame, threshold_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 保存视频帧
            if output_writer:
                output_writer.write(frame)
            
            # 显示视频
            if show_video:
                cv2.imshow('Fixed Wave Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 进度报告
            if frame_count % 200 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"进度: {progress:.1f}% - 挥手帧: {stats['frames_with_waves']}, 高宽高比: {stats['high_aspect_ratio_count']}")
    
    except KeyboardInterrupt:
        print("\n用户中断")
    
    # 清理资源
    cap.release()
    if output_writer:
        output_writer.release()
    if show_video:
        cv2.destroyAllWindows()
    
    # 输出统计
    print(f"\n=== 分析结果 ===")
    print(f"总帧数: {stats['total_frames']}")
    print(f"有人物帧数: {stats['frames_with_persons']}")
    print(f"检测到挥手帧数: {stats['frames_with_waves']}")
    print(f"高宽高比(>=0.7)次数: {stats['high_aspect_ratio_count']}")
    
    if stats['frames_with_persons'] > 0:
        wave_rate = (stats['frames_with_waves'] / stats['frames_with_persons']) * 100
        print(f"挥手检测率: {wave_rate:.1f}%")
    
    if output_path and os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024*1024)  # MB
        print(f"\n✅ 输出视频已保存: {output_path}")
        print(f"文件大小: {file_size:.1f} MB")
        
        # 验证文件是否可以打开
        test_cap = cv2.VideoCapture(output_path)
        if test_cap.isOpened():
            print("✅ 输出视频文件验证成功，可以正常打开")
            test_cap.release()
        else:
            print("❌ 输出视频文件验证失败")
    
    return True

def main():
    """主函数"""
    video_path = r"C:\c\wave_hands\test_videos_20250622_100947\B605_3min_20250622_100957.mov"
    
    print("=== 修复版挥手检测算法 ===")
    print("修复了以下问题：")
    print("1. 视频编码器兼容性问题")
    print("2. MOV文件处理问题")
    print("3. 帧率异常处理")
    print("4. 简化了算法逻辑")
    
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在: {video_path}")
        return
    
    success = analyze_video_with_fixed_encoding(
        video_path=video_path,
        show_video=True,
        save_output=True
    )
    
    if success:
        print("\n🎉 修复版算法测试完成！")
        print("\n主要改进:")
        print("- 使用AVI+XVID编码器（最兼容）")
        print("- 自动检测和修复帧率问题")
        print("- 简化了人物跟踪逻辑")
        print("- 降低了挥手持续时间要求(2帧)")

if __name__ == '__main__':
    main() 
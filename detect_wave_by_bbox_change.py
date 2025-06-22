import cv2
import yaml
import os
import sys
import time
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from collections import deque, defaultdict
import math

def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class PersonTracker:
    """人物跟踪器，用于跟踪同一个人的边界框变化"""
    
    def __init__(self, max_distance=100, max_frames_lost=10):
        self.tracks = {}  # track_id: track_info
        self.next_id = 0
        self.max_distance = max_distance  # 最大匹配距离
        self.max_frames_lost = max_frames_lost  # 最大丢失帧数
        
    def update(self, detections):
        """
        更新跟踪器
        detections: [(x1, y1, x2, y2, conf), ...]
        """
        # 计算检测框中心点
        detection_centers = []
        for det in detections:
            x1, y1, x2, y2, conf = det
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            detection_centers.append((center_x, center_y, det))
        
        # 更新现有轨迹的丢失计数
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['frames_lost'] += 1
            if self.tracks[track_id]['frames_lost'] > self.max_frames_lost:
                del self.tracks[track_id]
        
        # 匹配检测结果到现有轨迹
        matched_tracks = set()
        matched_detections = set()
        
        for i, (det_x, det_y, det) in enumerate(detection_centers):
            best_match = None
            best_distance = float('inf')
            
            for track_id, track_info in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                    
                # 计算与上一帧位置的距离
                last_x, last_y = track_info['last_center']
                distance = math.sqrt((det_x - last_x)**2 + (det_y - last_y)**2)
                
                if distance < self.max_distance and distance < best_distance:
                    best_distance = distance
                    best_match = track_id
            
            if best_match is not None:
                # 更新现有轨迹
                self.tracks[best_match]['last_center'] = (det_x, det_y)
                self.tracks[best_match]['last_bbox'] = det
                self.tracks[best_match]['frames_lost'] = 0
                matched_tracks.add(best_match)
                matched_detections.add(i)
        
        # 为未匹配的检测创建新轨迹
        for i, (det_x, det_y, det) in enumerate(detection_centers):
            if i not in matched_detections:
                self.tracks[self.next_id] = {
                    'last_center': (det_x, det_y),
                    'last_bbox': det,
                    'frames_lost': 0,
                    'created_frame': time.time()
                }
                self.next_id += 1
        
        return self.tracks

class WaveDetector:
    """基于边界框尺寸变化的挥手检测器"""
    
    def __init__(self, history_length=30, height_change_threshold=1.15, width_change_threshold=1.2, wave_duration_threshold=2.0):
        self.history_length = history_length  # 历史帧数
        self.height_change_threshold = height_change_threshold  # 高度变化阈值（倍数）
        self.width_change_threshold = width_change_threshold    # 宽度变化阈值（倍数）
        self.wave_duration_threshold = wave_duration_threshold  # 挥手持续时间阈值（秒）
        
        self.person_histories = defaultdict(lambda: deque(maxlen=history_length))  # track_id -> bbox_history
        self.wave_states = defaultdict(lambda: {'is_waving': False, 'wave_start_time': None, 'last_wave_time': None})
        
    def calculate_bbox_features(self, bbox):
        """计算边界框特征"""
        x1, y1, x2, y2, conf = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / height if height > 0 else 0
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        return {
            'width': width,
            'height': height, 
            'area': area,
            'aspect_ratio': aspect_ratio,
            'center': (center_x, center_y),
            'timestamp': time.time()
        }
    
    def detect_wave_motion(self, track_id, current_bbox):
        """检测挥手动作"""
        current_features = self.calculate_bbox_features(current_bbox)
        self.person_histories[track_id].append(current_features)
        
        # 需要足够的历史数据
        if len(self.person_histories[track_id]) < 10:
            return False, "insufficient_history"
        
        history = list(self.person_histories[track_id])
        current_time = current_features['timestamp']
        
        # 计算基准尺寸（使用历史数据的中位数）
        base_widths = [f['width'] for f in history[:-5]]  # 排除最近5帧
        base_heights = [f['height'] for f in history[:-5]]
        
        if not base_widths or not base_heights:
            return False, "insufficient_base_data"
        
        base_width = np.median(base_widths)
        base_height = np.median(base_heights)
        
        # 检查当前尺寸变化
        current_width = current_features['width']
        current_height = current_features['height']
        
        width_ratio = current_width / base_width if base_width > 0 else 1
        height_ratio = current_height / base_height if base_height > 0 else 1
        
        # 挥手特征判断
        is_height_increased = height_ratio > self.height_change_threshold
        is_width_increased = width_ratio > self.width_change_threshold
        
        # 检查最近几帧的变化趋势
        recent_frames = history[-5:]
        width_variance = 0
        height_variance = 0
        
        if len(recent_frames) >= 3:
            # 检查是否有连续的尺寸变化
            width_changes = []
            height_changes = []
            
            for i in range(1, len(recent_frames)):
                prev_w = recent_frames[i-1]['width']
                curr_w = recent_frames[i]['width']
                prev_h = recent_frames[i-1]['height']
                curr_h = recent_frames[i]['height']
                
                width_changes.append(curr_w / prev_w if prev_w > 0 else 1)
                height_changes.append(curr_h / prev_h if prev_h > 0 else 1)
            
            # 检查是否有明显的尺寸波动（挥手特征）
            width_variance = np.var(width_changes) if width_changes else 0
            height_variance = np.var(height_changes) if height_changes else 0
            
            has_size_fluctuation = width_variance > 0.01 or height_variance > 0.01
        else:
            has_size_fluctuation = False
        
        # 综合判断是否为挥手动作
        wave_indicators = {
            'height_increased': is_height_increased,
            'width_increased': is_width_increased,
            'size_fluctuation': has_size_fluctuation,
            'width_ratio': width_ratio,
            'height_ratio': height_ratio,
            'width_variance': width_variance,
            'height_variance': height_variance
        }
        
        # 挥手判断逻辑
        is_waving_frame = (
            (is_height_increased or is_width_increased) and  # 尺寸有明显增加
            has_size_fluctuation  # 有尺寸波动
        ) or (
            width_ratio > 1.1 and height_ratio > 1.1  # 两个维度都有适度增加
        )
        
        # 更新挥手状态
        wave_state = self.wave_states[track_id]
        
        if is_waving_frame:
            if not wave_state['is_waving']:
                wave_state['is_waving'] = True
                wave_state['wave_start_time'] = current_time
            wave_state['last_wave_time'] = current_time
        else:
            # 检查是否应该结束挥手状态
            if wave_state['is_waving'] and wave_state['last_wave_time']:
                time_since_last_wave = current_time - wave_state['last_wave_time']
                if time_since_last_wave > 1.0:  # 1秒内无挥手动作则结束
                    wave_state['is_waving'] = False
                    wave_state['wave_start_time'] = None
        
        # 判断是否为有效挥手（持续时间足够）
        is_valid_wave = False
        if wave_state['is_waving'] and wave_state['wave_start_time']:
            wave_duration = current_time - wave_state['wave_start_time']
            is_valid_wave = wave_duration >= self.wave_duration_threshold
        
        return is_valid_wave, wave_indicators

def analyze_video_with_bbox_wave_detection(video_path, output_dir=None, show_video=True, save_output=True):
    """
    使用边界框尺寸变化检测挥手动作
    """
    print(f"=== 基于边界框变化的挥手检测 ===")
    print(f"视频文件: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在")
        return False
    
    # 加载YOLO人物检测模型
    model = YOLO('yolov8n.pt')  # 使用预训练的COCO模型
    
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
    
    # 初始化跟踪器和检测器
    tracker = PersonTracker()
    wave_detector = WaveDetector()
    
    # 准备输出视频
    output_writer = None
    if save_output and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 根据原始文件格式选择输出格式和编码器
        original_ext = os.path.splitext(video_path)[1].lower()
        if original_ext == '.mov':
            output_path = os.path.join(output_dir, f"{video_name}_bbox_wave_{timestamp}.mov")
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # MOV兼容编码器
        else:
            output_path = os.path.join(output_dir, f"{video_name}_bbox_wave_{timestamp}.mp4")
            fourcc = cv2.VideoWriter.fourcc(*'XVID')  # 更兼容的编码器
        
        # 确保帧率有效
        if fps <= 0 or fps > 60:
            fps = 25.0
            print(f"警告：检测到异常帧率，设置为默认值 {fps}")
        
        output_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if output_writer.isOpened():
            print(f"保存结果到: {output_path}")
        else:
            print(f"警告：无法创建输出视频文件，尝试备用编码器...")
            # 尝试备用编码器
            fourcc_backup = cv2.VideoWriter.fourcc(*'mp4v')
            output_writer = cv2.VideoWriter(output_path, fourcc_backup, fps, (width, height))
            if not output_writer.isOpened():
                print("错误：无法创建输出视频文件")
                output_writer = None
    
    # 统计信息
    stats = {
        'total_frames': 0,
        'frames_with_persons': 0,
        'frames_with_waves': 0,
        'total_wave_detections': 0
    }
    
    frame_count = 0
    
    print("\n开始分析... (按 'q' 退出)")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            stats['total_frames'] = frame_count
            
            # 使用YOLO检测人物 (class_id=0 是人)
            results = model(frame, conf=0.5, verbose=False)[0]
            boxes = results.boxes
            
            # 提取人物检测结果
            person_detections = []
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if cls_id == 0:  # 人类
                        conf_score = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        person_detections.append((x1, y1, x2, y2, conf_score))
            
            if person_detections:
                stats['frames_with_persons'] += 1
            
            # 更新人物跟踪
            tracks = tracker.update(person_detections)
            
            # 检测挥手动作
            current_frame_has_wave = False
            wave_count = 0
            
            for track_id, track_info in tracks.items():
                if track_info['frames_lost'] == 0:  # 当前帧有检测结果
                    bbox = track_info['last_bbox']
                    is_waving, wave_info = wave_detector.detect_wave_motion(track_id, bbox)
                    
                    x1, y1, x2, y2, conf = bbox
                    
                    # 绘制人物边界框
                    if is_waving:
                        color = (0, 255, 0)  # 绿色 - 挥手
                        label = f"Person {track_id}: WAVING"
                        current_frame_has_wave = True
                        wave_count += 1
                        stats['total_wave_detections'] += 1
                    else:
                        color = (255, 0, 0)  # 蓝色 - 正常
                        label = f"Person {track_id}: Normal"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # 显示尺寸变化信息
                    if isinstance(wave_info, dict):
                        info_text = f"W:{wave_info.get('width_ratio', 0):.2f} H:{wave_info.get('height_ratio', 0):.2f}"
                        cv2.putText(frame, info_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if current_frame_has_wave:
                stats['frames_with_waves'] += 1
            
            # 在画面上显示统计信息
            info_text = f"Frame: {frame_count}/{total_frames} | Persons: {len([t for t in tracks.values() if t['frames_lost'] == 0])} | Waving: {wave_count}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 保存输出帧
            if output_writer:
                output_writer.write(frame)
            
            # 显示视频
            if show_video:
                cv2.imshow('Bbox Wave Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 进度报告
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"进度: {progress:.1f}% - 检测到挥手: {stats['total_wave_detections']} 次")
    
    except KeyboardInterrupt:
        print("\n用户中断分析")
    
    # 清理资源
    cap.release()
    if output_writer:
        output_writer.release()
    if show_video:
        cv2.destroyAllWindows()
    
    # 输出统计结果
    print(f"\n=== 分析结果 ===")
    print(f"总帧数: {stats['total_frames']}")
    print(f"有人物的帧数: {stats['frames_with_persons']}")
    print(f"检测到挥手的帧数: {stats['frames_with_waves']}")
    print(f"总挥手检测次数: {stats['total_wave_detections']}")
    
    if stats['frames_with_persons'] > 0:
        wave_rate = (stats['frames_with_waves'] / stats['frames_with_persons']) * 100
        print(f"挥手检测率: {wave_rate:.1f}%")
    
    if output_writer and save_output:
        print(f"分析结果已保存到: {output_path}")
    
    return True

def main():
    """主函数"""
    # 使用录制的视频进行测试
    video_path = r"C:\c\wave_hands\test_videos_20250622_100947\B605_3min_20250622_100957.mp4"
    
    print("=== 基于边界框尺寸变化的挥手检测 ===")
    print("原理：检测人物边界框的高度和宽度变化来判断挥手动作")
    print("- 挥手时手臂抬起 → 高度增加")
    print("- 向侧面挥动 → 宽度增加")
    print("- 结合时间序列分析判断动作持续性")
    
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在")
        return
    
    # 开始分析
    success = analyze_video_with_bbox_wave_detection(
        video_path=video_path,
        output_dir="bbox_wave_analysis",
        show_video=True,
        save_output=True
    )
    
    if success:
        print("\n✓ 分析完成！")
        print("\n这种方法的优势：")
        print("1. 无需重新训练模型")
        print("2. 基于成熟的人物检测")
        print("3. 利用挥手的几何特征")
        print("4. 实时性能好")
        
        print("\n可以进一步优化的方向：")
        print("1. 调整尺寸变化阈值")
        print("2. 优化时间窗口参数")
        print("3. 加入更多几何特征（如纵横比变化）")
        print("4. 结合姿态估计进一步提高准确性")

if __name__ == '__main__':
    main() 
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

class PersonTracker:
    """简化版人物跟踪器"""
    
    def __init__(self, max_distance=100, max_frames_lost=10):
        self.tracks = {}
        self.next_id = 0
        self.max_distance = max_distance
        self.max_frames_lost = max_frames_lost
        
    def update(self, detections):
        detection_centers = []
        for det in detections:
            x1, y1, x2, y2, conf = det
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            detection_centers.append((center_x, center_y, det))
        
        # 更新现有轨迹
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['frames_lost'] += 1
            if self.tracks[track_id]['frames_lost'] > self.max_frames_lost:
                del self.tracks[track_id]
        
        # 匹配检测结果
        matched_tracks = set()
        matched_detections = set()
        
        for i, (det_x, det_y, det) in enumerate(detection_centers):
            best_match = None
            best_distance = float('inf')
            
            for track_id, track_info in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                    
                last_x, last_y = track_info['last_center']
                distance = math.sqrt((det_x - last_x)**2 + (det_y - last_y)**2)
                
                if distance < self.max_distance and distance < best_distance:
                    best_distance = distance
                    best_match = track_id
            
            if best_match is not None:
                self.tracks[best_match]['last_center'] = (det_x, det_y)
                self.tracks[best_match]['last_bbox'] = det
                self.tracks[best_match]['frames_lost'] = 0
                matched_tracks.add(best_match)
                matched_detections.add(i)
        
        # 创建新轨迹
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

class ImprovedWaveDetector:
    """改进的挥手检测器，带详细调试信息"""
    
    def __init__(self, history_length=20):
        self.history_length = history_length
        self.person_histories = defaultdict(lambda: deque(maxlen=history_length))
        self.debug_info = defaultdict(list)
        
    def calculate_bbox_features(self, bbox):
        x1, y1, x2, y2, conf = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / height if height > 0 else 0
        
        return {
            'width': width,
            'height': height,
            'area': area,
            'aspect_ratio': aspect_ratio,
            'timestamp': time.time()
        }
    
    def detect_wave_motion_debug(self, track_id, current_bbox, frame_count):
        """带调试信息的挥手检测"""
        current_features = self.calculate_bbox_features(current_bbox)
        self.person_histories[track_id].append(current_features)
        
        # 调试信息
        debug_info = {
            'frame': frame_count,
            'track_id': track_id,
            'current_width': current_features['width'],
            'current_height': current_features['height'],
            'current_aspect_ratio': current_features['aspect_ratio'],
            'history_length': len(self.person_histories[track_id])
        }
        
        # 需要至少5帧历史数据
        if len(self.person_histories[track_id]) < 5:
            debug_info['result'] = 'insufficient_history'
            debug_info['is_waving'] = False
            return False, debug_info
        
        history = list(self.person_histories[track_id])
        
        # 方法1：基于固定基准的简单检测
        # 计算基准尺寸（前几帧的平均值）
        base_frames = history[:min(5, len(history)//2)]
        if base_frames:
            base_width = np.mean([f['width'] for f in base_frames])
            base_height = np.mean([f['height'] for f in base_frames])
            base_aspect_ratio = np.mean([f['aspect_ratio'] for f in base_frames])
        else:
            debug_info['result'] = 'no_base_frames'
            debug_info['is_waving'] = False
            return False, debug_info
        
        # 计算变化比例
        width_ratio = current_features['width'] / base_width if base_width > 0 else 1
        height_ratio = current_features['height'] / base_height if base_height > 0 else 1
        aspect_ratio_change = current_features['aspect_ratio'] / base_aspect_ratio if base_aspect_ratio > 0 else 1
        
        debug_info.update({
            'base_width': base_width,
            'base_height': base_height,
            'base_aspect_ratio': base_aspect_ratio,
            'width_ratio': width_ratio,
            'height_ratio': height_ratio,
            'aspect_ratio_change': aspect_ratio_change
        })
        
        # 方法2：检查最近几帧的变化幅度
        recent_frames = history[-5:]
        if len(recent_frames) >= 3:
            widths = [f['width'] for f in recent_frames]
            heights = [f['height'] for f in recent_frames]
            aspect_ratios = [f['aspect_ratio'] for f in recent_frames]
            
            width_std = np.std(widths)
            height_std = np.std(heights)
            aspect_std = np.std(aspect_ratios)
            
            # 计算变化幅度（标准差 / 平均值）
            width_variation = width_std / np.mean(widths) if np.mean(widths) > 0 else 0
            height_variation = height_std / np.mean(heights) if np.mean(heights) > 0 else 0
            aspect_variation = aspect_std / np.mean(aspect_ratios) if np.mean(aspect_ratios) > 0 else 0
            
            debug_info.update({
                'width_variation': width_variation,
                'height_variation': height_variation,
                'aspect_variation': aspect_variation
            })
        else:
            width_variation = height_variation = aspect_variation = 0
            debug_info.update({
                'width_variation': 0,
                'height_variation': 0,
                'aspect_variation': 0
            })
        
        # 简化的挥手判断逻辑
        # 条件1：宽度明显增加（放宽阈值）
        width_increased = width_ratio > 1.1  # 10%增加
        
        # 条件2：高度增加
        height_increased = height_ratio > 1.05  # 5%增加
        
        # 条件3：宽高比明显变化
        aspect_changed = aspect_ratio_change > 1.15 or aspect_ratio_change < 0.85
        
        # 条件4：有明显的尺寸变化（波动）
        has_variation = (width_variation > 0.05 or height_variation > 0.05 or 
                        aspect_variation > 0.05)
        
        # 条件5：当前宽高比明显偏离正常人体比例
        # 正常人体宽高比大约是 0.4-0.6，挥手时可能达到 0.8-1.2
        abnormal_aspect_ratio = current_features['aspect_ratio'] > 0.3
        
        debug_info.update({
            'width_increased': width_increased,
            'height_increased': height_increased,
            'aspect_changed': aspect_changed,
            'has_variation': has_variation,
            'abnormal_aspect_ratio': abnormal_aspect_ratio
        })
        
        # 综合判断（放宽条件）
        is_waving = (
            abnormal_aspect_ratio or  # 主要依据：异常宽高比
            (width_increased and has_variation) or  # 宽度增加+变化
            (height_increased and aspect_changed) or  # 高度增加+比例变化
            (width_increased and height_increased)  # 两个维度都增加
        )
        
        debug_info['is_waving'] = is_waving
        debug_info['result'] = 'waving' if is_waving else 'normal'
        
        # 保存调试信息
        self.debug_info[track_id].append(debug_info)
        
        return is_waving, debug_info

def debug_wave_detection(video_path, max_frames=500):
    """调试挥手检测算法"""
    print(f"=== 调试挥手检测算法 ===")
    print(f"视频: {video_path}")
    print(f"最大分析帧数: {max_frames}")
    
    if not os.path.exists(video_path):
        print("视频文件不存在")
        return
    
    # 加载模型
    model = YOLO('yolov8n.pt')
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 初始化
    tracker = PersonTracker()
    detector = ImprovedWaveDetector()
    
    frame_count = 0
    all_debug_info = []
    
    print("\n开始分析...")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 检测人物
        results = model(frame, conf=0.5, verbose=False)[0]
        boxes = results.boxes
        
        person_detections = []
        if boxes is not None:
            for box in boxes:
                if int(box.cls[0]) == 0:  # 人类
                    conf_score = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_detections.append((x1, y1, x2, y2, conf_score))
        
        # 跟踪和检测
        tracks = tracker.update(person_detections)
        
        for track_id, track_info in tracks.items():
            if track_info['frames_lost'] == 0:
                bbox = track_info['last_bbox']
                is_waving, debug_info = detector.detect_wave_motion_debug(track_id, bbox, frame_count)
                all_debug_info.append(debug_info)
        
        # 每100帧打印一次进度
        if frame_count % 100 == 0:
            wave_count = sum(1 for info in all_debug_info if info['is_waving'])
            print(f"进度: {frame_count} 帧, 检测到挥手: {wave_count} 次")
    
    cap.release()
    
    # 分析调试信息
    print(f"\n=== 调试分析结果 ===")
    print(f"总分析帧数: {len(all_debug_info)}")
    
    wave_frames = [info for info in all_debug_info if info['is_waving']]
    normal_frames = [info for info in all_debug_info if not info['is_waving']]
    
    print(f"挥手帧数: {len(wave_frames)}")
    print(f"正常帧数: {len(normal_frames)}")
    print(f"挥手检测率: {len(wave_frames)/len(all_debug_info)*100:.1f}%")
    
    # 统计分析
    if all_debug_info:
        all_width_ratios = [info['width_ratio'] for info in all_debug_info if 'width_ratio' in info]
        all_height_ratios = [info['height_ratio'] for info in all_debug_info if 'height_ratio' in info]
        all_aspect_ratios = [info['current_aspect_ratio'] for info in all_debug_info if 'current_aspect_ratio' in info]
        
        print(f"\n=== 尺寸变化统计 ===")
        print(f"宽度比例范围: {min(all_width_ratios):.2f} - {max(all_width_ratios):.2f}")
        print(f"高度比例范围: {min(all_height_ratios):.2f} - {max(all_height_ratios):.2f}")
        print(f"宽高比范围: {min(all_aspect_ratios):.2f} - {max(all_aspect_ratios):.2f}")
        
        print(f"平均宽度比例: {np.mean(all_width_ratios):.2f}")
        print(f"平均高度比例: {np.mean(all_height_ratios):.2f}")
        print(f"平均宽高比: {np.mean(all_aspect_ratios):.2f}")
    
    # 显示一些具体的样本
    print(f"\n=== 挥手样本分析 ===")
    for i, info in enumerate(wave_frames[:5]):  # 显示前5个挥手样本
        print(f"样本 {i+1}:")
        print(f"  帧 {info['frame']}: 宽高比={info['current_aspect_ratio']:.2f}, "
              f"宽度比={info['width_ratio']:.2f}, 高度比={info['height_ratio']:.2f}")
    
    print(f"\n=== 高宽高比样本 ===")
    high_aspect_samples = [info for info in all_debug_info if info.get('current_aspect_ratio', 0) > 0.3]
    print(f"宽高比>0.7的帧数: {len(high_aspect_samples)}")
    
    for i, info in enumerate(high_aspect_samples[:10]):  # 显示前10个高宽高比样本
        status = "挥手" if info['is_waving'] else "未识别"
        print(f"  帧 {info['frame']}: 宽高比={info['current_aspect_ratio']:.2f}, "
              f"宽度比={info.get('width_ratio', 0):.2f}, {status}")
    
    # 建议阈值调整
    if high_aspect_samples:
        high_aspect_waving = [info for info in high_aspect_samples if info['is_waving']]
        high_aspect_not_waving = [info for info in high_aspect_samples if not info['is_waving']]
        
        print(f"\n=== 建议 ===")
        print(f"高宽高比样本中，{len(high_aspect_waving)}/{len(high_aspect_samples)} 被识别为挥手")
        print(f"建议：")
        print(f"1. 如果宽高比>0.7的情况大多是挥手，可以简化判断逻辑")
        print(f"2. 当前算法可能过于复杂，可以直接基于宽高比阈值判断")
        print(f"3. 考虑降低各种变化的阈值要求")

def main():
    video_path = r"C:\c\wave_hands\test_videos_20250622_100947\B605_3min_20250622_100957.mov"
    debug_wave_detection(video_path, max_frames=1000)

if __name__ == '__main__':
    main() 
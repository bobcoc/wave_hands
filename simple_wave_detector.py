import cv2
import os
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from collections import deque, defaultdict
import math

class SimplePersonTracker:
    """简化的人物跟踪器"""
    
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
                    'frames_lost': 0
                }
                self.next_id += 1
        
        return self.tracks

class SimpleWaveDetector:
    """基于宽高比的简化挥手检测器"""
    
    def __init__(self, 
                 normal_aspect_ratio_max=0.6,  # 正常人体宽高比上限
                 wave_aspect_ratio_min=0.7,    # 挥手时宽高比下限
                 history_length=10,             # 历史帧数
                 min_wave_duration=3):          # 最小挥手持续帧数
        
        self.normal_aspect_ratio_max = normal_aspect_ratio_max
        self.wave_aspect_ratio_min = wave_aspect_ratio_min
        self.history_length = history_length
        self.min_wave_duration = min_wave_duration
        
        self.person_histories = defaultdict(lambda: deque(maxlen=history_length))
        self.wave_counters = defaultdict(int)  # 连续挥手帧计数
        
    def detect_wave_simple(self, track_id, bbox):
        """简化的挥手检测：主要基于宽高比"""
        x1, y1, x2, y2, conf = bbox
        width = x2 - x1
        height = y2 - y1
        
        if height <= 0:
            return False, {'aspect_ratio': 0, 'is_waving': False, 'reason': 'invalid_height'}
        
        aspect_ratio = width / height
        
        # 记录历史
        self.person_histories[track_id].append(aspect_ratio)
        
        # 判断逻辑
        is_current_frame_wave = aspect_ratio >= self.wave_aspect_ratio_min
        
        if is_current_frame_wave:
            self.wave_counters[track_id] += 1
        else:
            self.wave_counters[track_id] = 0  # 重置计数器
        
        # 需要连续几帧都是挥手状态才认为是真正的挥手
        is_sustained_wave = self.wave_counters[track_id] >= self.min_wave_duration
        
        # 获取历史统计
        history = list(self.person_histories[track_id])
        avg_aspect_ratio = np.mean(history) if history else aspect_ratio
        max_aspect_ratio = max(history) if history else aspect_ratio
        
        debug_info = {
            'aspect_ratio': aspect_ratio,
            'avg_aspect_ratio': avg_aspect_ratio,
            'max_aspect_ratio': max_aspect_ratio,
            'wave_counter': self.wave_counters[track_id],
            'is_current_wave': is_current_frame_wave,
            'is_sustained_wave': is_sustained_wave,
            'width': width,
            'height': height
        }
        
        return is_sustained_wave, debug_info

def analyze_video_simple_wave_detection(video_path, output_dir=None, show_video=True, save_output=True):
    """使用简化算法进行挥手检测"""
    print(f"=== 简化版挥手检测 ===")
    print(f"检测原理：人物宽高比 >= 0.7 且持续3帧以上 = 挥手")
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
    
    # 初始化检测器
    tracker = SimplePersonTracker()
    wave_detector = SimpleWaveDetector(
        normal_aspect_ratio_max=0.6,
        wave_aspect_ratio_min=0.7,   # 可以调整这个阈值
        min_wave_duration=3
    )
    
    # 准备输出视频
    output_writer = None
    if save_output and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 根据原始文件格式选择输出格式
        original_ext = os.path.splitext(video_path)[1].lower()
        if original_ext == '.mov':
            output_path = os.path.join(output_dir, f"{video_name}_simple_wave_{timestamp}.mov")
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        else:
            output_path = os.path.join(output_dir, f"{video_name}_simple_wave_{timestamp}.avi")
            fourcc = cv2.VideoWriter.fourcc(*'XVID')
        
        # 确保帧率有效
        if fps <= 0 or fps > 60:
            fps = 25.0
            
        output_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if output_writer and output_writer.isOpened():
            print(f"保存结果到: {output_path}")
        else:
            print(f"警告：无法创建输出视频文件")
            output_writer = None
    
    # 统计信息
    stats = {
        'total_frames': 0,
        'frames_with_persons': 0,
        'frames_with_waves': 0,
        'total_wave_detections': 0,
        'high_aspect_ratio_frames': 0  # 宽高比>0.7的帧数
    }
    
    frame_count = 0
    
    print(f"\n开始分析... (按 'q' 退出)")
    print(f"检测阈值: 宽高比 >= {wave_detector.wave_aspect_ratio_min}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            stats['total_frames'] = frame_count
            
            # 使用YOLO检测人物
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
                    is_waving, wave_info = wave_detector.detect_wave_simple(track_id, bbox)
                    
                    x1, y1, x2, y2, conf = bbox
                    aspect_ratio = wave_info['aspect_ratio']
                    
                    # 统计高宽高比帧
                    if aspect_ratio >= 0.7:
                        stats['high_aspect_ratio_frames'] += 1
                    
                    # 绘制边界框
                    if is_waving:
                        color = (0, 255, 0)  # 绿色 - 挥手
                        label = f"Person {track_id}: WAVING"
                        current_frame_has_wave = True
                        wave_count += 1
                        stats['total_wave_detections'] += 1
                    elif wave_info['is_current_wave']:
                        color = (0, 255, 255)  # 黄色 - 可能挥手（持续时间不够）
                        label = f"Person {track_id}: MAYBE WAVE"
                    else:
                        color = (255, 0, 0)  # 蓝色 - 正常
                        label = f"Person {track_id}: Normal"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # 显示宽高比信息
                    ratio_text = f"Ratio: {aspect_ratio:.2f} ({wave_info['wave_counter']})"
                    cv2.putText(frame, ratio_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if current_frame_has_wave:
                stats['frames_with_waves'] += 1
            
            # 在画面上显示统计信息
            info_text = f"Frame: {frame_count}/{total_frames} | Persons: {len([t for t in tracks.values() if t['frames_lost'] == 0])} | Waving: {wave_count}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示检测阈值
            threshold_text = f"Wave Threshold: Aspect Ratio >= {wave_detector.wave_aspect_ratio_min}"
            cv2.putText(frame, threshold_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 保存输出帧
            if output_writer:
                output_writer.write(frame)
            
            # 显示视频
            if show_video:
                cv2.imshow('Simple Wave Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 进度报告
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"进度: {progress:.1f}% - 挥手: {stats['total_wave_detections']}, 高宽高比: {stats['high_aspect_ratio_frames']}")
    
    except KeyboardInterrupt:
        print("\n用户中断分析")
    
    # 清理资源
    cap.release()
    if output_writer:
        output_writer.release()
    if show_video:
        cv2.destroyAllWindows()
    
    # 输出统计结果
    print(f"\n=== 简化算法分析结果 ===")
    print(f"总帧数: {stats['total_frames']}")
    print(f"有人物的帧数: {stats['frames_with_persons']}")
    print(f"检测到挥手的帧数: {stats['frames_with_waves']}")
    print(f"总挥手检测次数: {stats['total_wave_detections']}")
    print(f"高宽高比(>=0.7)帧数: {stats['high_aspect_ratio_frames']}")
    
    if stats['frames_with_persons'] > 0:
        wave_rate = (stats['frames_with_waves'] / stats['frames_with_persons']) * 100
        high_aspect_rate = (stats['high_aspect_ratio_frames'] / stats['frames_with_persons']) * 100
        print(f"挥手检测率: {wave_rate:.1f}%")
        print(f"高宽高比出现率: {high_aspect_rate:.1f}%")
        
        if stats['high_aspect_ratio_frames'] > stats['total_wave_detections']:
            print(f"\n💡 发现: 有 {stats['high_aspect_ratio_frames'] - stats['total_wave_detections']} 个高宽高比帧未被识别为挥手")
            print("建议: 可以降低 min_wave_duration 参数或调整 wave_aspect_ratio_min 阈值")
    
    if output_writer and save_output:
        print(f"\n✅ 分析视频已保存: {output_path}")
    
    return True

def main():
    """主函数"""
    video_path = r"C:\c\wave_hands\test_videos_20250622_100947\B605_3min_20250622_100957.mov"
    
    print("=== 简化版挥手检测算法 ===")
    print("核心思路: 如果人物宽高比 >= 0.7 且持续3帧以上，则判定为挥手")
    print("这个方法直接利用了您观察到的'挥手时宽高比明显异常'特征")
    
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在: {video_path}")
        return
    
    # 开始分析
    success = analyze_video_simple_wave_detection(
        video_path=video_path,
        output_dir="simple_wave_analysis",
        show_video=True,
        save_output=True
    )
    
    if success:
        print("\n🎉 简化算法测试完成！")
        print("\n可调整的参数：")
        print("- wave_aspect_ratio_min: 当前0.7，可以调整为0.6或0.8")
        print("- min_wave_duration: 当前3帧，可以调整为1或2")
        print("- normal_aspect_ratio_max: 当前0.6，用于对比")

if __name__ == '__main__':
    main() 
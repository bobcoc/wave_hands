import cv2
import os
import numpy as np
import mediapipe as mp
from datetime import datetime
from collections import deque, defaultdict
import math
import yaml

def load_config(config_path='config.yaml'):
    """加载配置文件 - 与原版完全一致"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class MediaPipeWaveDetector:
    """MediaPipe Holistic 挥手检测器 - 替代YOLO+DeepSORT方案"""
    
    def __init__(self, config):
        # 初始化MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1  # 平衡性能和精度
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # 从配置读取参数（与原版保持一致）
        self.teacher_height_threshold = int(config.get('teacher_height_threshold', 400))
        self.wave_change_threshold = int(config.get('wave_change_threshold', 3))
        self.confidence = float(config.get('confidence', 0.7))
        self.device = config.get('device', 'cpu')
        
        # 滑动窗口长度（与原版一致：75帧 = 3秒）
        self.window_len = 75
        
        # 人员跟踪数据（替代DeepSORT的功能）
        self.person_trackers = {}
        self.next_person_id = 1
        
        # 人员识别阈值
        self.person_distance_threshold = 150  # 像素距离
        
    def calculate_person_height_pixels(self, landmarks, frame_height):
        """基于关键点计算人体实际像素高度（替代YOLO边界框高度）"""
        try:
            # 使用鼻子到脚踝的距离估算身高
            nose = landmarks[self.mp_holistic.PoseLandmark.NOSE]
            left_ankle = landmarks[self.mp_holistic.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[self.mp_holistic.PoseLandmark.RIGHT_ANKLE]
            
            # 取双脚平均位置
            avg_ankle_y = (left_ankle.y + right_ankle.y) / 2
            height_normalized = abs(avg_ankle_y - nose.y)
            height_pixels = height_normalized * frame_height
            
            return height_pixels
        except:
            return 0.0
    
    def calculate_person_center(self, landmarks):
        """计算人体中心点（用于人员追踪，替代DeepSORT的ID分配）"""
        try:
            # 使用肩部中点作为人体中心
            left_shoulder = landmarks[self.mp_holistic.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER]
            
            center_x = (left_shoulder.x + right_shoulder.x) / 2
            center_y = (left_shoulder.y + right_shoulder.y) / 2
            
            return (center_x, center_y)
        except:
            return None
    
    def get_or_assign_person_id(self, center, frame_width, frame_height):
        """根据人体中心位置分配或获取人员ID（替代DeepSORT追踪）"""
        if center is None:
            return None
            
        center_pixels = (center[0] * frame_width, center[1] * frame_height)
        
        # 查找最近的现有追踪目标
        min_distance = float('inf')
        closest_id = None
        
        for person_id, tracker_data in self.person_trackers.items():
            last_center = tracker_data.get('last_center')
            if last_center:
                distance = math.sqrt(
                    (center_pixels[0] - last_center[0])**2 + 
                    (center_pixels[1] - last_center[1])**2
                )
                if distance < min_distance and distance < self.person_distance_threshold:
                    min_distance = distance
                    closest_id = person_id
        
        if closest_id:
            # 更新现有追踪目标
            self.person_trackers[closest_id]['last_center'] = center_pixels
            return closest_id
        else:
            # 创建新的追踪目标
            new_id = self.next_person_id
            self.next_person_id += 1
            self.person_trackers[new_id] = {
                'last_center': center_pixels,
                'right_arm_angle_history': deque(maxlen=self.window_len),
                'frame_count': 0
            }
            return new_id
    
    def calculate_arm_angle(self, shoulder, elbow, wrist):
        """计算右手臂角度（替代边界框尺寸变化检测）"""
        try:
            # 计算上臂和前臂向量
            upper_arm = [elbow.x - shoulder.x, elbow.y - shoulder.y]
            lower_arm = [wrist.x - elbow.x, wrist.y - elbow.y]
            
            # 计算向量夹角
            dot_product = upper_arm[0] * lower_arm[0] + upper_arm[1] * lower_arm[1]
            norm_upper = math.sqrt(upper_arm[0]**2 + upper_arm[1]**2)
            norm_lower = math.sqrt(lower_arm[0]**2 + lower_arm[1]**2)
            
            if norm_upper == 0 or norm_lower == 0:
                return None
                
            cos_angle = dot_product / (norm_upper * norm_lower)
            cos_angle = max(-1.0, min(1.0, cos_angle))  # 数值稳定性
            angle = math.degrees(math.acos(cos_angle))
            
            return angle
        except:
            return None
    
    def detect_angle_changes(self, angle_history, change_threshold=15):
        """检测角度变化次数（对应原版的detect_dimension_changes函数）"""
        if len(angle_history) < 2:
            return 0, []
        
        changes = []
        change_count = 0
        
        angles = list(angle_history)
        for i in range(1, len(angles)):
            prev_angle = angles[i-1]
            curr_angle = angles[i]
            angle_change = abs(curr_angle - prev_angle)
            
            if angle_change >= change_threshold:
                change_count += 1
                changes.append(('angle_change', i, prev_angle, curr_angle, angle_change))
        
        return change_count, changes
    
    def process_frame(self, frame, frame_idx):
        """处理单帧（主要检测逻辑，对应原版main函数的核心部分）"""
        frame_disp = frame.copy()
        height, width = frame.shape[:2]
        
        # MediaPipe处理
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)
        
        detected_persons = []
        waving_ids = set()
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 计算人体属性
            height_pixels = self.calculate_person_height_pixels(landmarks, height)
            center = self.calculate_person_center(landmarks)
            
            # 教师过滤（对应原版的TEACHER_HEIGHT_THRESHOLD检查）
            if height_pixels >= self.teacher_height_threshold:
                person_id = self.get_or_assign_person_id(center, width, height)
                
                if person_id is not None:
                    tracker_data = self.person_trackers[person_id]
                    tracker_data['frame_count'] += 1
                    
                    # 检测右手挥手动作
                    try:
                        right_shoulder = landmarks[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                        right_elbow = landmarks[self.mp_holistic.PoseLandmark.RIGHT_ELBOW]
                        right_wrist = landmarks[self.mp_holistic.PoseLandmark.RIGHT_WRIST]
                        
                        right_angle = self.calculate_arm_angle(right_shoulder, right_elbow, right_wrist)
                        
                        if right_angle is not None:
                            tracker_data['right_arm_angle_history'].append(right_angle)
                            
                            # 检测角度变化（对应原版的detect_dimension_changes）
                            change_count, changes = self.detect_angle_changes(
                                tracker_data['right_arm_angle_history'], 
                                change_threshold=15
                            )
                            
                            # 挥手判定（对应原版的波动检测逻辑）
                            is_waving = change_count >= self.wave_change_threshold
                            if is_waving:
                                waving_ids.add(person_id)
                            
                            # 构建检测结果数据
                            person_info = {
                                'person_id': person_id,
                                'height_pixels': height_pixels,
                                'right_angle': right_angle,
                                'angle_history_length': len(tracker_data['right_arm_angle_history']),
                                'change_count': change_count,
                                'changes': changes,
                                'is_waving': is_waving,
                                'center': center,
                                'landmarks': landmarks
                            }
                            detected_persons.append(person_info)
                            
                    except Exception as e:
                        print(f"  Error processing person {person_id}: {e}")
        
        # 绘制检测结果
        self.draw_detection_results(frame_disp, results, detected_persons, waving_ids, frame_idx, width, height)
        
        return frame_disp, detected_persons, waving_ids
    
    def draw_detection_results(self, frame, results, detected_persons, waving_ids, frame_idx, width, height):
        """绘制检测结果（对应原版的可视化部分）"""
        # 绘制MediaPipe关键点
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        
        if results.right_hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_draw.DrawingSpec(color=(66,245,170), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_draw.DrawingSpec(color=(66,170,245), thickness=2, circle_radius=2)
            )
        
        # 绘制人员检测信息（对应原版的边界框和标签）
        for person_info in detected_persons:
            person_id = person_info['person_id']
            center = person_info['center']
            is_waving = person_info['is_waving']
            
            if center:
                display_x = int(center[0] * width)
                display_y = int(center[1] * height)
                
                # 状态颜色（绿色=挥手，红色=正常）
                color = (0, 255, 0) if is_waving else (255, 0, 0)
                status = "WAVING" if is_waving else "NORMAL"
                
                # 绘制人员中心标记
                cv2.circle(frame, (display_x, display_y), 15, color, -1)
                cv2.circle(frame, (display_x, display_y), 18, (255, 255, 255), 2)
                
                # 显示详细信息（对应原版的标签格式）
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                
                labels = [
                    f"ID{person_id}: {status} (Teacher)",
                    f"Height:{person_info['height_pixels']:.0f}px",
                    f"Angle:{person_info['right_angle']:.1f}° Changes:{person_info['change_count']}",
                    f"History:{person_info['angle_history_length']}"
                ]
                
                # 计算文本背景尺寸
                max_width = 0
                for label in labels:
                    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                    max_width = max(max_width, text_width)
                
                # 绘制白色文本背景
                bg_height = len(labels) * 25 + 10
                cv2.rectangle(frame, (display_x-5, display_y-bg_height-30), 
                             (display_x+max_width+10, display_y+5), (255, 255, 255), -1)
                cv2.rectangle(frame, (display_x-5, display_y-bg_height-30), 
                             (display_x+max_width+10, display_y+5), (0, 0, 0), 1)
                
                # 绘制黑色文本
                for i, label in enumerate(labels):
                    y_pos = display_y - bg_height + i * 20
                    cv2.putText(frame, label, (display_x, y_pos), font, font_scale, (0, 0, 0), thickness)
        
        # 全局信息显示（对应原版的统计信息）
        info_lines = [
            f"Frame: {frame_idx}",
            f"Teachers detected: {len(detected_persons)}",
            f"Waving detected: {len(waving_ids)}",
            f"Filter: Height>{self.teacher_height_threshold}px, AngleChange>15°"
        ]
        
        # 绘制全局信息背景和文本
        global_font = cv2.FONT_HERSHEY_SIMPLEX
        global_font_scale = 0.6
        global_thickness = 2
        
        max_global_width = 0
        for info_line in info_lines:
            (text_width, text_height), _ = cv2.getTextSize(info_line, global_font, global_font_scale, global_thickness)
            max_global_width = max(max_global_width, text_width)
        
        global_bg_height = len(info_lines) * 25 + 20
        cv2.rectangle(frame, (10, 10), (max_global_width + 30, global_bg_height), (255, 255, 255), -1)
        cv2.rectangle(frame, (10, 10), (max_global_width + 30, global_bg_height), (0, 0, 0), 2)
        
        for i, info_line in enumerate(info_lines):
            cv2.putText(frame, info_line, (15, 35 + i*20), global_font, global_font_scale, (0, 0, 0), global_thickness)

def main():
    """主函数 - 与原版流程完全对应"""
    # 读取配置（与原版一致）
    config = load_config()
    video_path = config.get('input_video', 'input.mp4')
    output_dir = config.get('alarm_dir', 'alarms')
    os.makedirs(output_dir, exist_ok=True)
    alarm_duration = int(config.get('alarm_duration', 3))
    name = os.path.splitext(os.path.basename(video_path))[0]

    # 初始化MediaPipe检测器（替代YOLO+DeepSORT）
    detector = MediaPipeWaveDetector(config)

    # 打开视频（与原版一致）
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"=== MediaPipe Holistic Wave Detection Analysis ===")
    print(f"Video file: {video_path}")
    print(f"Video properties: {width}x{height}, {fps:.1f}fps, {total_frames} frames")
    print(f"Sliding window: {detector.window_len} frames (3 seconds), threshold: {detector.wave_change_threshold}")
    print(f"Right hand angle change detection: >15° threshold")
    print(f"Analysis started... (press 'q' to quit)")

    # 创建输出文件（对应原版的日志文件）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 角度数据日志（替代原版的aspect_ratios日志）
    angle_log_path = os.path.join(output_dir, f"{name}_arm_angles_{timestamp}.txt")
    angle_log_file = open(angle_log_path, 'w', encoding='utf-8')
    angle_log_file.write("# Arm angle data log\n")
    angle_log_file.write("# Format: frame_id,person_id,right_arm_angle,height_pixels,center_x,center_y\n")
    
    # 处理后视频输出（与原版一致）
    output_video_dir = "mediapipe_wave_analysis"
    os.makedirs(output_video_dir, exist_ok=True)
    output_video_path = os.path.join(output_video_dir, f"{name}_mediapipe_wave_{timestamp}.mp4")
    
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    output_video_writer = cv2.VideoWriter(output_video_path, fourcc, fps if fps > 0 else 25, (width, height))
    
    print(f"Arm angle data will be saved to: {angle_log_path}")
    print(f"Processed video will be saved to: {output_video_path}")

    # 主处理循环（与原版结构一致）
    frame_buffer = deque(maxlen=detector.window_len)
    alarm_active = False
    frame_idx = 0
    no_person_count = 0
    NO_PERSON_WARN_THRESHOLD = 50

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        frame_buffer.append(frame.copy())
        
        # 处理当前帧
        processed_frame, detected_persons, waving_ids = detector.process_frame(frame, frame_idx)
        
        # 输出检测结果（对应原版的打印输出）
        print(f"Frame {frame_idx} detection results:")
        if detected_persons:
            for person_info in detected_persons:
                person_id = person_info['person_id']
                print(f"  Person {person_id}: Height={person_info['height_pixels']:.0f}px, "
                      f"Angle={person_info['right_angle']:.1f}°, Changes={person_info['change_count']}")
                
                # 显示变化详情
                if person_info['changes']:
                    print(f"    Changes: ", end="")
                    for change_type, frame_pos, prev_val, curr_val, change_magnitude in person_info['changes']:
                        actual_frame = frame_idx - person_info['angle_history_length'] + frame_pos
                        print(f"Frame{actual_frame}: {prev_val:.1f}→{curr_val:.1f} (Δ{change_magnitude:.1f}°) ", end="")
                    print()
                
                if person_info['is_waving']:
                    print(f"    *** ID{person_id} DETECTED AS WAVING! ***")
                
                # 记录数据到文件
                center = person_info['center']
                if center:
                    angle_log_file.write(f"{frame_idx},{person_id},{person_info['right_angle']:.2f},"
                                       f"{person_info['height_pixels']:.0f},{center[0]:.3f},{center[1]:.3f}\n")
                    angle_log_file.flush()
            
            no_person_count = 0
        else:
            print("  No teachers detected")
            no_person_count += 1
            if no_person_count == NO_PERSON_WARN_THRESHOLD:
                print(f"⚠️ Warning: No teacher detected for {NO_PERSON_WARN_THRESHOLD} consecutive frames!")

        # 保存处理后的视频帧
        output_video_writer.write(processed_frame)

        # 报警机制（与原版完全一致）
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

        # 进度显示（与原版一致）
        if frame_idx % 200 == 0:
            if total_frames > 0:
                print(f"Progress: {frame_idx}/{total_frames} frames ({(frame_idx/total_frames)*100:.1f}%)")

        # 实时显示
        cv2.imshow('MediaPipe Wave Detection', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 清理资源（与原版一致）
    cap.release()
    output_video_writer.release()
    angle_log_file.close()
    cv2.destroyAllWindows()
    
    print(f"\n=== Processing Complete ===")
    print(f"Arm angle data saved: {angle_log_path}")
    print(f"Processed video saved: {output_video_path}")
    
    # 文件验证（与原版一致）
    if os.path.exists(angle_log_path):
        file_size = os.path.getsize(angle_log_path) / 1024  # KB
        print(f"Arm angle data file size: {file_size:.1f} KB")
    
    if os.path.exists(output_video_path):
        file_size = os.path.getsize(output_video_path) / (1024*1024)  # MB
        print(f"Processed video file size: {file_size:.1f} MB")

if __name__ == '__main__':
    main() 
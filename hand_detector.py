import cv2
import numpy as np
from ultralytics import YOLO
from mediapipe_hand import mediapipe_detect_hands, analyze_hand_pose

class HandDetector:
    """统一的手势识别类，使用YOLO检测手掌位置，然后用MediaPipe或YOLO提取关键点"""
    
    def __init__(self, detector='mediapipe', weights='weight/best.pt', confidence=0.2, 
                 device='cpu', font_scale=1.2, font_thickness=2, margin_ratio=0.02):
        """
        初始化手势识别器
        :param detector: 'mediapipe' 或 'yolo'，选择关键点检测器
        :param weights: YOLO权重文件路径（用于手掌检测）
        :param confidence: YOLO置信度阈值
        :param device: 设备，'cpu' 或 'cuda'
        :param font_scale: 显示文字大小
        :param font_thickness: 显示文字粗细
        :param margin_ratio: 边框扩展比例，相对于检测框的大小
        """
        self.detector_type = detector
        self.confidence = confidence
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.margin_ratio = margin_ratio
        
        # 骨架连线定义
        self.connections = [
            (0,1),(1,2),(2,3),(3,4),    # 大拇指
            (0,5),(5,6),(6,7),(7,8),    # 食指
            (0,9),(9,10),(10,11),(11,12), # 中指
            (0,13),(13,14),(14,15),(15,16), # 无名指
            (0,17),(17,18),(18,19),(19,20)  # 小指
        ]
        
        # 初始化YOLO模型（用于手掌检测）
        self.palm_detector = YOLO(weights, task='detect')
        self.palm_detector.to(device)
        
        # 如果使用YOLO做关键点检测，需要第二个模型
        if detector == 'yolo':
            self.keypoint_detector = YOLO(weights, task='detect')
            self.keypoint_detector.to(device)
        else:
            self.keypoint_detector = None
    
    def detect_frame(self, frame):
        """
        检测单帧图像中的手势
        :return: List[Dict]，每个手的信息包含：
                - keypoints: 关键点坐标
                - handedness: 左右手
                - is_palm_up: 是否手掌朝上
                - bbox: 边界框 (x1,y1,x2,y2)
        """
        hands_info = []
        
        # 1. 使用YOLO检测手掌位置
        palm_results = self.palm_detector(frame, conf=self.confidence, verbose=False)[0]
        
        # 2. 对每个检测到的手掌区域进行关键点检测
        for box in palm_results.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:  # 手掌类
                # 获取边界框并扩大区域
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])  # 获取置信度值
                
                # 计算动态边距
                box_width = x2 - x1
                box_height = y2 - y1
                margin_x = int(box_width * self.margin_ratio)
                margin_y = int(box_height * self.margin_ratio)
                
                # 扩展检测框
                x1 = max(0, x1 - margin_x)
                y1 = max(0, y1 - margin_y)
                x2 = min(frame.shape[1], x2 + margin_x)
                y2 = min(frame.shape[0], y2 + margin_y)
                
                # 提取ROI
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                
                # 在ROI中检测关键点
                if self.detector_type == 'mediapipe':
                    roi_hands = mediapipe_detect_hands(roi)
                    # 调整关键点坐标到原图坐标系
                    for hand in roi_hands:
                        adjusted_keypoints = []
                        for kpt_x, kpt_y in hand['keypoints']:
                            adj_x = int(kpt_x + x1)
                            adj_y = int(kpt_y + y1)
                            adjusted_keypoints.append((adj_x, adj_y))
                        hand['keypoints'] = adjusted_keypoints
                        hand['bbox'] = (x1, y1, x2, y2)
                        hand['confidence'] = conf  # 添加置信度
                        hands_info.append(hand)
                else:  # yolo
                    roi_results = self.keypoint_detector(roi, conf=self.confidence, verbose=False)[0]
                    if hasattr(roi_results, 'keypoints') and roi_results.keypoints is not None:
                        for kpts in roi_results.keypoints.xy:
                            if len(kpts) == 21:
                                # 调整关键点坐标到原图坐标系
                                adjusted_keypoints = [(int(x + x1), int(y + y1)) for x, y in kpts]
                                pose_info = analyze_hand_pose(adjusted_keypoints)
                                hands_info.append({
                                    'keypoints': adjusted_keypoints,
                                    'handedness': pose_info['handedness'],
                                    'is_palm_up': pose_info['is_palm_up'],
                                    'bbox': (x1, y1, x2, y2),
                                    'confidence': conf  # 添加置信度
                                })
        
        return hands_info
    
    def draw_hand_info(self, frame, hands_info):
        """
        在图像上绘制手势信息
        :param frame: 输入图像
        :param hands_info: detect_frame的返回结果
        :return: 绘制了手势信息的图像
        """
        output = frame.copy()
        
        for hand in hands_info:
            keypoints = hand['keypoints']
            if len(keypoints) == 21:
                # 获取手的信息
                hand_type = hand['handedness']
                is_palm_up = hand['is_palm_up']
                bbox = hand['bbox']
                
                # 画检测框
                x1, y1, x2, y2 = bbox
                cv2.rectangle(output, (x1, y1), (x2, y2), (0,255,0), 2)
                
                # 在检测框右边显示置信度
                conf = hand.get('confidence', 0.0)
                conf_text = f"{conf:.2f}"
                (cw, ch), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                            self.font_scale, self.font_thickness)
                y_mid = (y1 + y2) // 2
                cv2.rectangle(output, (x2+5, y_mid-ch), (x2+5+cw+8, y_mid+ch+8), (0,0,0), -1)
                cv2.putText(output, conf_text, (x2+7, y_mid+ch+2), cv2.FONT_HERSHEY_SIMPLEX, 
                           self.font_scale, (255,255,255), self.font_thickness)
                
                # 生成标签
                side = "Palm" if is_palm_up else "Back"
                label = f"{hand_type} {side}"
                
                # 画骨架
                for conn in self.connections:
                    pt1 = keypoints[conn[0]]
                    pt2 = keypoints[conn[1]]
                    cv2.line(output, pt1, pt2, (255,0,0), 2)
                
                # 画关键点
                for x, y in keypoints:
                    cv2.circle(output, (x, y), 3, (0,255,0), -1)
                
                # 画标签
                x0, y0 = keypoints[0]  # 手腕位置
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                            self.font_scale, self.font_thickness)
                text_bg_top = max(0, y0 - 20 - th)
                text_bg_bottom = y0 - 4
                cv2.rectangle(output, (x0, text_bg_top), 
                            (x0+tw+8, text_bg_bottom), (0,0,0), -1)
                cv2.putText(output, label, (x0+4, text_bg_bottom-4), 
                           cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 
                           (255,255,255), self.font_thickness)
        
        return output
    
    def process_frame(self, frame):
        """
        处理单帧图像（检测+绘制）
        :return: (处理后的图像, 手势信息)
        """
        hands_info = self.detect_frame(frame)
        output = self.draw_hand_info(frame, hands_info)
        return output, hands_info 
import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands

# 进程级全局变量，每个进程独立的MediaPipe实例
_process_mediapipe_hands = None
_hands_call_count = 0
_hands_reset_interval = 1000  # 每1000次调用重置一次，防止内存泄漏


def get_mediapipe_hands_instance(max_num_hands=2, min_detection_confidence=0.5):
    """
    获取或创建当前进程的MediaPipe Hands实例（进程安全）
    每个进程有独立的实例，避免每次调用都初始化
    """
    global _process_mediapipe_hands, _hands_call_count
    
    # 定期重置防止状态污染
    if _hands_call_count >= _hands_reset_interval and _process_mediapipe_hands is not None:
        try:
            _process_mediapipe_hands.close()
        except:
            pass
        _process_mediapipe_hands = None
        _hands_call_count = 0
        pid = os.getpid()
        print(f"[PID:{pid}] MediaPipe实例已重置（达到{_hands_reset_interval}次调用）")
    
    # 创建新实例
    if _process_mediapipe_hands is None:
        pid = os.getpid()
        print(f"[PID:{pid}] 创建 MediaPipe Hands 实例...")
        _process_mediapipe_hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence
        )
        print(f"[PID:{pid}] MediaPipe Hands 实例创建完成")
    
    _hands_call_count += 1
    return _process_mediapipe_hands

class HandLandmark:
    """手部关键点的通用格式，用于统一YOLO和MediaPipe的输出"""
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

def calculate_palm_normal(landmarks, hand_type):
    """计算手掌法向量，用于判断手掌朝向
    landmarks: 包含x,y,z属性的关键点列表
    hand_type: "Left" or "Right"，用于调整法向量方向
    """
    # 用食指(5)、中指(9)和无名指(13)的起点构建平面
    v1 = np.array([landmarks[5].x - landmarks[9].x,
                   landmarks[5].y - landmarks[9].y,
                   landmarks[5].z - landmarks[9].z])
    v2 = np.array([landmarks[13].x - landmarks[9].x,
                   landmarks[13].y - landmarks[9].y,
                   landmarks[13].z - landmarks[9].z])
    # 计算法向量
    normal = np.cross(v1, v2)
    # 归一化
    normal = normal / np.linalg.norm(normal)
    
    # 根据手的类型调整法向量方向
    if hand_type == "Right":
        normal = -normal  # 右手需要翻转法向量
        
    return normal

def determine_hand_type(keypoints):
    """根据关键点判断左右手
    keypoints: [(x,y), ...]
    """
    # 通过拇指(4)和小指(20)的x坐标判断
    thumb_tip_x = keypoints[4][0]
    pinky_tip_x = keypoints[20][0]
    return "Right" if thumb_tip_x < pinky_tip_x else "Left"

def analyze_hand_pose(keypoints, hand_type=None, with_z=False):
    """分析手的姿态，可用于YOLO或MediaPipe的关键点
    keypoints: [(x,y), ...] 或 [(x,y,z), ...]
    hand_type: 如果为None，会自动判断
    with_z: 关键点是否包含z坐标
    """
    if hand_type is None:
        hand_type = determine_hand_type(keypoints)
    
    # 转换为HandLandmark格式
    landmarks = []
    for kp in keypoints:
        if with_z and len(kp) > 2:
            landmarks.append(HandLandmark(kp[0], kp[1], kp[2]))
        else:
            landmarks.append(HandLandmark(kp[0], kp[1]))
    
    # 计算手掌朝向
    normal = calculate_palm_normal(landmarks, hand_type)
    is_palm_up = normal[2] < 0
    
    return {
        'handedness': hand_type,
        'is_palm_up': is_palm_up
    }

def mediapipe_detect_hands(frame, max_num_hands=2, min_detection_confidence=0.5):
    """
    使用MediaPipe检测手部关键点和分类信息
    返回：List of dict，每个dict包含：
    - keypoints: List[(x, y)]，21个关键点坐标
    - handedness: str，"Left" or "Right"
    - is_palm_up: bool，True表示手掌朝上/前
    """
    # 复用MediaPipe实例（关键优化！避免每次都初始化）
    hands = get_mediapipe_hands_instance(max_num_hands, min_detection_confidence)
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    hands_info = []
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # 获取关键点坐标
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.append((int(lm.x * frame.shape[1]), 
                                int(lm.y * frame.shape[0]),
                                lm.z))
            
            # 获取手的分类（左/右手），需要翻转因为是镜像
            hand_type = "Right" if handedness.classification[0].label == "Left" else "Left"
            
            # 分析手势
            pose_info = analyze_hand_pose(keypoints, hand_type, with_z=True)
            
            hands_info.append({
                'keypoints': [(x, y) for x, y, _ in keypoints],
                'handedness': pose_info['handedness'],
                'is_palm_up': pose_info['is_palm_up']
            })
    
    return hands_info 
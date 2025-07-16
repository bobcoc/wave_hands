import cv2
import numpy as np
from ultralytics import YOLO

# MediaPipe骨架连线
MEDIAPIPE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),    # 大拇指
    (0,5),(5,6),(6,7),(7,8),    # 食指
    (0,9),(9,10),(10,11),(11,12), # 中指
    (0,13),(13,14),(14,15),(15,16), # 无名指
    (0,17),(17,18),(18,19),(19,20)  # 小指
]

def load_yolo_model(weights, device='cpu'):
    """加载YOLO模型"""
    model = YOLO(weights, task='detect')
    model.to(device)
    return model

def detect_hands(model, frame, conf=0.5):
    """对单帧进行手部检测，返回检测结果"""
    results = model(frame, conf=conf, verbose=False)[0]
    return results

def classify_hand_pose(keypoints):
    """
    输入: keypoints (21,2) ndarray，顺序与MediaPipe一致
    输出: 'Left Palm', 'Left Back', 'Right Palm', 'Right Back', 'Unknown'
    """
    if keypoints is None or len(keypoints) != 21:
        return "Unknown"
    kpts = np.array(keypoints)
    thumb_tip_x = kpts[4,0]
    pinky_tip_x = kpts[20,0]
    if thumb_tip_x < pinky_tip_x:
        hand_type = "Right"
    else:
        hand_type = "Left"
    v1 = kpts[5] - kpts[0]
    v2 = kpts[17] - kpts[0]
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    if hand_type == "Right":
        is_palm = cross > 0
    else:
        is_palm = cross < 0
    if is_palm:
        side = "Palm"
    else:
        side = "Back"
    return f"{hand_type} {side}"

def draw_hand_overlay(frame, results, level, mediapipe_connections=MEDIAPIPE_CONNECTIONS, font_scale=1.2, font_thickness=2):
    """绘制检测框、关键点、标签等，可自定义字体大小和粗细"""
    out_frame = frame.copy()
    if level == 0 or results is None:
        return out_frame
    for i, box in enumerate(getattr(results, 'boxes', [])):
        cls_id = int(box.cls[0])
        if cls_id == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0,255,0), 2)
            hand_label = "waving"
            if level == 2 and hasattr(results, 'keypoints') and results.keypoints is not None:
                kpts = results.keypoints.xy[i]
                if isinstance(kpts, np.ndarray):
                    hand_label = classify_hand_pose(kpts)
                elif hasattr(kpts, 'numpy'):  # 兼容 torch.Tensor
                    hand_label = classify_hand_pose(kpts.numpy())
                else:
                    hand_label = "Unknown"
            else:
                hand_label = "Unknown"
            text = hand_label
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            # 文本上移到框上方，且加大背景高度
            text_bg_top = max(0, y1 - 20 - th)
            text_bg_bottom = y1 - 4
            cv2.rectangle(out_frame, (x1, text_bg_top), (x1+tw+8, text_bg_bottom), (0,0,0), -1)
            cv2.putText(out_frame, text, (x1+4, text_bg_bottom-4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), font_thickness)
            conf_text = f"{float(box.conf[0]):.2f}"
            (cw, ch), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            y_mid = (y1 + y2) // 2
            cv2.rectangle(out_frame, (x2+5, y_mid-ch), (x2+5+cw+8, y_mid+ch+8), (0,0,0), -1)
            cv2.putText(out_frame, conf_text, (x2+7, y_mid+ch+2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), font_thickness)
            if level == 2 and hasattr(results, 'keypoints') and results.keypoints is not None:
                kpts = results.keypoints.xy[i]
                for idx, (x, y) in enumerate(kpts):
                    cv2.circle(out_frame, (int(x), int(y)), 2, (0,255,0), -1)
                if mediapipe_connections:
                    for conn in mediapipe_connections:
                        pt1 = kpts[conn[0]]
                        pt2 = kpts[conn[1]]
                        cv2.line(out_frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255,0,0), 1)
    return out_frame 
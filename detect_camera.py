import cv2
from hand_detection_core import MEDIAPIPE_CONNECTIONS
from mediapipe_hand import mediapipe_detect_hands, analyze_hand_pose
from ultralytics import YOLO

def detect_camera(detector='yolo', weights='weight/best.pt', confidence=0.5, device='cpu',
                 overlay_level=2, camera_id=0, font_scale=1.2, font_thickness=2):
    """
    手势检测主函数
    detector: 'mediapipe' 或 'yolo'，选择检测器
    """
    # 如果使用YOLO，加载模型
    model = None
    if detector == 'yolo':
        model = YOLO(weights, task='detect')
        model.to(device)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"无法打开摄像头 {camera_id}")
        return

    print(f"使用 {detector} 检测，按 q 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break

        # 检测手部
        if detector == 'mediapipe':
            hands_info = mediapipe_detect_hands(frame)
        else:  # yolo
            results = model(frame, conf=confidence, verbose=False)[0]
            hands_info = []
            if hasattr(results, 'keypoints') and results.keypoints is not None:
                for i, kpts in enumerate(results.keypoints.xy):
                    if len(kpts) == 21:  # 确保有21个关键点
                        # 分析手势
                        pose_info = analyze_hand_pose(kpts)
                        hands_info.append({
                            'keypoints': [(int(x), int(y)) for x, y in kpts],
                            'handedness': pose_info['handedness'],
                            'is_palm_up': pose_info['is_palm_up']
                        })

        # 画关键点、骨架和标签
        for hand in hands_info:
            keypoints = hand['keypoints']
            if len(keypoints) == 21:
                # 获取手的信息
                hand_type = hand['handedness']
                is_palm_up = hand['is_palm_up']
                
                # 生成标签
                side = "Palm" if is_palm_up else "Back"
                label = f"{hand_type} {side}"
                
                # 画骨架
                for conn in MEDIAPIPE_CONNECTIONS:
                    pt1 = keypoints[conn[0]]
                    pt2 = keypoints[conn[1]]
                    cv2.line(frame, pt1, pt2, (255,0,0), 2)
                
                # 画关键点
                for x, y in keypoints:
                    cv2.circle(frame, (x, y), 3, (0,255,0), -1)
                
                # 画标签
                x0, y0 = keypoints[0]  # 手腕位置
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                text_bg_top = max(0, y0 - 20 - th)
                text_bg_bottom = y0 - 4
                cv2.rectangle(frame, (x0, text_bg_top), (x0+tw+8, text_bg_bottom), (0,0,0), -1)
                cv2.putText(frame, label, (x0+4, text_bg_bottom-4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), font_thickness)

        cv2.imshow(f'Hand Detection ({detector})', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 默认使用 MediaPipe
    detect_camera(detector='mediapipe')
    
    # 如果要使用 YOLO，取消下面的注释
    # detect_camera(detector='yolo', weights='weight/best.pt')

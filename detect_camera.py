import cv2
from hand_detector import HandDetector

def detect_camera(detector='yolo', weights='weight/best.pt', confidence=0.5, device='cpu',
                 overlay_level=2, camera_id=0, font_scale=1.2, font_thickness=2):
    """
    手势检测主函数
    detector: 'mediapipe' 或 'yolo'，选择检测器
    """
    # 初始化检测器
    hand_detector = HandDetector(
        detector=detector,
        weights=weights,
        confidence=confidence,
        device=device,
        font_scale=font_scale,
        font_thickness=font_thickness
    )

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

        # 处理当前帧
        output, hands_info = hand_detector.process_frame(frame)
        
        # 显示结果
        cv2.imshow(f'Hand Detection ({detector})', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 默认使用 MediaPipe
    detect_camera(detector='mediapipe',weights='weight/best.pt')
    
    # 如果要使用 YOLO，取消下面的注释
    # detect_camera(detector='yolo', weights='weight/best.pt')

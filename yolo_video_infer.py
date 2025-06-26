import cv2
import yaml
import os
from datetime import datetime
from ultralytics import YOLO

# 1. 读取配置文件
def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # 2. 加载配置
    config = load_config()
    input_video = config.get('input_video', 'input.mp4')
    weights = config.get('weights', 'weight/best.pt')
    confidence = float(config.get('confidence', 0.5))
    device = config.get('device', 'cpu')

    # 3. 加载YOLO模型
    try:
        model = YOLO(weights, task='detect')
        model.to(device)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 4. 打开输入视频
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"无法打开输入视频: {input_video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 5. 创建输出目录和输出视频
    output_video_dir = "yolo_wave_analysis"
    os.makedirs(output_video_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(input_video))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_video_path = os.path.join(output_video_dir, f"{name}_yolo_wave_{timestamp}.mp4")
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    output_video_writer = cv2.VideoWriter(output_video_path, fourcc, fps if fps > 0 else 25, (width, height))

    print(f"=== YOLO Wave Detection Analysis ===")
    print(f"Input video: {input_video}")
    print(f"Output video: {output_video_path}")
    print(f"Video properties: {width}x{height}, {fps:.1f}fps, {total_frames} frames")
    print(f"Confidence threshold: {confidence}")
    print(f"Device: {device}")
    print(f"Processing...")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        # 6. 推理
        results = model(frame, conf=confidence, verbose=False)[0]
        # 7. 可视化检测框和置信度
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # 只画palm/waving类别（单类，id=0）
            if cls_id == 0:
                color_box = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_box, 2)
                # 置信度
                conf_text = f"{conf_score:.2f}"
                (cw, ch), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                y_mid = (y1 + y2) // 2
                cv2.rectangle(frame, (x2+5, y_mid-ch), (x2+5+cw+4, y_mid+ch+4), (0,0,0), -1)
                cv2.putText(frame, conf_text, (x2+7, y_mid+ch), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        # 8. 写入输出视频
        output_video_writer.write(frame)
        # 9. 进度显示
        if frame_idx % 100 == 0:
            if total_frames > 0:
                print(f"Progress: {frame_idx}/{total_frames} frames ({(frame_idx/total_frames)*100:.1f}%)")
            else:
                print(f"Progress: {frame_idx} frames processed")

    # 10. 资源释放
    cap.release()
    output_video_writer.release()
    print(f"\n=== Processing Complete ===")
    print(f"Output video saved: {output_video_path}")
    if os.path.exists(output_video_path):
        file_size = os.path.getsize(output_video_path) / (1024*1024)
        print(f"Output video file size: {file_size:.1f} MB")

if __name__ == '__main__':
    main() 
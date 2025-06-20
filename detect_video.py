import cv2
import yaml
from ultralytics import YOLO
import os
import sys
import time
from collections import deque
from multiprocessing import Process

# 读取配置文件
def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def play_video_window(video_path, window_name='Alarm', wait=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开报警视频: {video_path}")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(window_name, frame)
        if cv2.waitKey(wait) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyWindow(window_name)

def worker(stream_cfg, config):
    import cv2
    import time
    from collections import deque
    from ultralytics import YOLO
    import os

    name = stream_cfg.get('name', 'noname')
    url = stream_cfg.get('url')
    weights = config.get('weights', 'weight/best.pt')
    confidence = float(config.get('confidence', 0.5))
    names = config.get('names', ['wave', 'nowave'])
    device = config.get('device', 'cpu')
    alarm_dir = config.get('alarm_dir', 'alarms')
    alarm_classes = config.get('alarm_classes', [0, 1])
    alarm_duration = int(config.get('alarm_duration', 3))
    detect_interval_ms = int(config.get('detect_interval_ms', 500))
    cooldown_seconds = int(config.get('cooldown_seconds', 60))

    if not os.path.exists(alarm_dir):
        os.makedirs(alarm_dir)

    model = YOLO(weights, task='detect')
    model.to(device)

    color_map = [(0,255,0), (0,0,255), (255,0,0), (255,255,0), (0,255,255)]
    state = 'idle'  # idle, active, cooldown
    cooldown_until = 0
    frame_buffer = None
    alarm_flags = None
    wave_counts = None
    nowave_counts = None
    alarm_buf_len = None
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    print(f"[{name}] 进程启动，流地址: {url}")
    while True:
        now = time.time()
        if state == 'cooldown':
            if now < cooldown_until:
                time.sleep(1)
                continue
            else:
                print(f"[{name}] 冷却结束，恢复检测")
                state = 'idle'
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print(f"[{name}] 无法打开视频流: {url}，5秒后重试")
            time.sleep(5)
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        alarm_buf_len = int(alarm_duration * (fps if fps > 0 else 25))
        frame_buffer = deque(maxlen=alarm_buf_len)
        alarm_flags = deque(maxlen=alarm_buf_len)
        wave_counts = deque(maxlen=alarm_buf_len)
        nowave_counts = deque(maxlen=alarm_buf_len)
        print(f"[{name}] 开始检测，fps={fps}, 分辨率={width}x{height}")
        while True:
            if state == 'idle':
                time.sleep(detect_interval_ms / 1000.0)
            ret, frame = cap.read()
            if not ret:
                print(f"[{name}] 读取帧失败，重启流")
                cap.release()
                break
            results = model(frame, conf=confidence, verbose=False)[0]
            boxes = results.boxes
            wave_count = 0
            nowave_count = 0
            for box in boxes:
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                if conf_score < confidence:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = color_map[cls_id % len(color_map)]
                label = f"{names[cls_id]} {conf_score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                if cls_id == 0:
                    wave_count += 1
                if cls_id == 1:
                    nowave_count += 1
            has_alarm_class = (wave_count > 0 or nowave_count > 0)
            frame_buffer.append(frame.copy())
            alarm_flags.append(has_alarm_class)
            wave_counts.append(wave_count)
            nowave_counts.append(nowave_count)
            # 状态切换逻辑
            if state == 'idle':
                if has_alarm_class:
                    print(f"[{name}] 检测到wave/nowave，进入active状态")
                    state = 'active'
            elif state == 'active':
                if not has_alarm_class:
                    print(f"[{name}] 未检测到wave/nowave，回到idle状态")
                    state = 'idle'
                    continue
                # 报警逻辑
                if len(alarm_flags) == alarm_buf_len:
                    if all(alarm_flags):
                        total_wave = sum(wave_counts)
                        total_nowave = sum(nowave_counts)
                        if total_nowave == 0:
                            ratio = float('inf') if total_wave > 0 else 0
                        else:
                            ratio = total_wave / total_nowave
                        if ratio >= 3:
                            ts = time.strftime('%Y%m%d_%H%M%S')
                            alarm_path = os.path.join(alarm_dir, f'{name}_{ts}.mp4')
                            alarm_writer = cv2.VideoWriter(alarm_path, fourcc, fps if fps > 0 else 25, (width, height))
                            for f in frame_buffer:
                                alarm_writer.write(f)
                            alarm_writer.release()
                            print(f"[{name}] !!! 报警：连续{alarm_duration}秒wave/nowave，wave:nowave={total_wave}:{total_nowave}，报警片段: {alarm_path}")
                            play_video_window(alarm_path, window_name=f'ALARM-{name}', wait=int(1000/(fps if fps > 0 else 25)))
                            state = 'cooldown'
                            cooldown_until = time.time() + cooldown_seconds
                            break
        cap.release()

def main():
    config = load_config()
    streams = config.get('streams', [])
    if not streams:
        print('配置文件未找到streams字段或为空')
        sys.exit(1)
    processes = []
    for stream_cfg in streams:
        p = Process(target=worker, args=(stream_cfg, config))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

if __name__ == '__main__':
    main() 
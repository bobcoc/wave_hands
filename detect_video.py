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

# 创建支持硬件解码的VideoCapture对象
def create_video_capture_with_hwdecode(url, config, name):
    """
    创建支持硬件解码的VideoCapture对象
    """
    video_decode_config = config.get('video_decode', {})
    use_hardware_decode = video_decode_config.get('use_hardware_decode', False)
    buffer_size = video_decode_config.get('buffer_size', 1)
    decode_backend = video_decode_config.get('decode_backend', 'ffmpeg')
    fallback_to_software = video_decode_config.get('fallback_to_software', True)
    rtsp_transport = video_decode_config.get('rtsp_transport', 'tcp')
    
    cap = None
    
    # 根据传输协议修改URL
    if rtsp_transport.lower() == 'tcp' and 'rtsp://' in url:
        # 对于TCP传输，可以在FFmpeg选项中设置
        modified_url = url
    else:
        modified_url = url
    
    if use_hardware_decode:
        try:
            print(f"[{name}] 尝试使用硬件解码器，传输协议: {rtsp_transport}")
            # 使用FFmpeg后端
            if decode_backend.lower() == 'ffmpeg':
                cap = cv2.VideoCapture(modified_url, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(modified_url)
            
            if cap.isOpened():
                # 设置缓冲区大小
                cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
                # 尝试设置硬件解码相关参数
                try:
                    # 设置解码器优化参数
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('H', '2', '6', '4'))
                    # 设置更多容错参数
                    cap.set(cv2.CAP_PROP_FPS, 25)  # 限制FPS减少缓冲压力
                    # 设置RTSP传输协议为TCP（更稳定但延迟更高）
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30000)  # 30秒超时
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 30000)  # 30秒读取超时
                except Exception as e:
                    print(f"[{name}] 硬件解码参数设置警告: {e}")
                
                print(f"[{name}] 硬件解码器初始化成功")
                return cap
            else:
                print(f"[{name}] 硬件解码器初始化失败")
                if cap:
                    cap.release()
                    cap = None
        except Exception as e:
            print(f"[{name}] 硬件解码器创建异常: {e}")
            if cap:
                cap.release()
                cap = None
    
    # 软件解码备用方案
    if cap is None and fallback_to_software:
        try:
            print(f"[{name}] 回退到软件解码器...")
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
                print(f"[{name}] 软件解码器初始化成功")
            else:
                print(f"[{name}] 软件解码器也初始化失败")
        except Exception as e:
            print(f"[{name}] 软件解码器创建异常: {e}")
    
    return cap

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
    device = config.get('device', 'cpu')
    alarm_dir = config.get('alarm_dir', 'alarms')
    alarm_duration = int(config.get('alarm_duration', 3))
    cooldown_seconds = int(config.get('cooldown_seconds', 60))
    idle_detect_interval = int(config.get('idle_detect_interval', 5))  # 新增，idle下每N帧检测一次

    if not os.path.exists(alarm_dir):
        os.makedirs(alarm_dir)

    model = YOLO(weights, task='detect')
    model.to(device)

    color_box = (0, 255, 0)
    color_kpt = (0, 255, 255)
    color_line = (255, 255, 255)
    state = 'idle'  # idle, active, cooldown
    cooldown_until = 0
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    print(f"[{name}] 进程启动，流地址: {url}")

    cap = None
    force_reconnect = False
    idle_frame_counter = 0
    alarm_buf_len = 75  # 3秒*25帧，后续可根据fps动态调整
    mediapipe_connections = [  # MediaPipe手部骨架连线
        (0,1),(1,2),(2,3),(3,4),    # 大拇指
        (0,5),(5,6),(6,7),(7,8),    # 食指
        (0,9),(9,10),(10,11),(11,12), # 中指
        (0,13),(13,14),(14,15),(15,16), # 无名指
        (0,17),(17,18),(18,19),(19,20)  # 小指
    ]
    # 新增：定时重连相关变量
    reconnect_interval = int(config.get('reconnect_interval', 120))  # 秒
    last_reconnect_time = time.time()
    # 新增：全局帧计数
    global_frame_counter = 0

    while True:
        # 定时重连逻辑
        if time.time() - last_reconnect_time >= reconnect_interval:
            print(f"[{name}] 定期重连：已运行{reconnect_interval}秒，主动重新连接")
            if cap:
                cap.release()
            cap = None
            last_reconnect_time = time.time()
            continue

        if state == 'cooldown':
            if time.time() < cooldown_until:
                time.sleep(0.1)
                continue
            else:
                print(f"[{name}] 状态切换: cooldown -> idle (冷却结束)")
                state = 'idle'
                idle_frame_counter = 0

        if not cap or not cap.isOpened():
            cap = create_video_capture_with_hwdecode(url, config, name)
            if not cap or not cap.isOpened():
                print(f"[{name}] 无法打开视频流: {url}，5秒后重试")
                if cap:
                    cap.release()
                cap = None
                time.sleep(5)
                continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps > 0:
                alarm_buf_len = int(alarm_duration * fps)
            else:
                alarm_buf_len = 75
            print(f"[{name}] 连接成功，fps={fps}, 分辨率={width}x{height}, 报警缓冲区长度={alarm_buf_len}")
            last_reconnect_time = time.time()  # 连接成功后重置重连计时

        if state == 'idle':
            # 跳帧检测
            for _ in range(idle_detect_interval-1):
                cap.grab()
            ret, frame = cap.read()
            if not ret:
                print(f"[{name}] idle模式读取帧失败，重连")
                cap.release()
                cap = None
                continue
            results = model(frame, conf=confidence, verbose=False)[0]
            # 在画面上显示所有检测到的目标
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if cls_id == 0:
                    # palm类别，左上角显示waving，右侧中间显示置信度，均为黑底白字
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_box, 2)
                    # waving文字
                    text = 'waving'
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x1, y1-10-th), (x1+tw+4, y1-10+4), (0,0,0), -1)
                    cv2.putText(frame, text, (x1+2, y1-10+th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                    # 置信度分数
                    conf_text = f"{conf_score:.2f}"
                    (cw, ch), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    y_mid = (y1 + y2) // 2
                    cv2.rectangle(frame, (x2+5, y_mid-ch), (x2+5+cw+4, y_mid+ch+4), (0,0,0), -1)
                    cv2.putText(frame, conf_text, (x2+7, y_mid+ch), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                # 其它类别不显示任何文字
            palm_found = False
            palm_roi = None  # 先初始化
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:
                    palm_found = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    palm_roi = (x1, y1, x2, y2)
                    break
            global_frame_counter += 1
            print(f"[{name}] [idle] 已处理帧数: {global_frame_counter}")
            if palm_found:
                last_palm_roi = palm_roi  # 立即保存初始ROI，确保active分支可用
                print(f"[{name}] 状态切换: idle -> active (检测到palm)")
                state = 'active'
                active_buffer = []
                palm_frame_count = 0
                active_frame_counter = 0
                continue  # 进入active
            continue

        if state == 'active':
            # ROI区域由上次检测到的palm框决定，每次检测到palm时更新
            if 'last_palm_roi' not in locals() or last_palm_roi is None:
                print(f"[{name}] 警告：active状态下last_palm_roi无效，回退idle")
                state = 'idle'
                continue
            x1, y1, x2, y2 = last_palm_roi
            roi_x1 = max(0, x1-300)
            roi_y1 = max(0, y1-300)
            roi_x2 = min(width, x2+300)
            roi_y2 = min(height, y2+300)
            ret, frame = cap.read()
            if not ret:
                print(f"[{name}] active模式读取帧失败，重连")
                cap.release()
                cap = None
                state = 'idle'
                continue
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            results = model(roi, conf=confidence, verbose=False)[0]
            palm_in_this_frame = False
            new_palm_roi = None
            for i, box in enumerate(results.boxes):
                cls_id = int(box.cls[0])
                if cls_id == 0:
                    palm_in_this_frame = True
                    rx1, ry1, rx2, ry2 = map(int, box.xyxy[0])
                    x1g, y1g, x2g, y2g = rx1+roi_x1, ry1+roi_y1, rx2+roi_x1, ry2+roi_y1
                    new_palm_roi = (x1g, y1g, x2g, y2g)
                    cv2.rectangle(frame, (x1g, y1g), (x2g, y2g), color_box, 2)
                    text = 'waving'
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x1g, y1g-10-th), (x1g+tw+4, y1g-10+4), (0,0,0), -1)
                    cv2.putText(frame, text, (x1g+2, y1g-10+th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                    conf_text = f"{float(box.conf[0]):.2f}"
                    (cw, ch), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    y_mid = (y1g + y2g) // 2
                    cv2.rectangle(frame, (x2g+5, y_mid-ch), (x2g+5+cw+4, y_mid+ch+4), (0,0,0), -1)
                    cv2.putText(frame, conf_text, (x2g+7, y_mid+ch), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                    if hasattr(results, 'keypoints') and results.keypoints is not None:
                        kpts = results.keypoints.xy[i]
                        for idx, (x, y) in enumerate(kpts):
                            cv2.circle(frame, (int(x)+roi_x1, int(y)+roi_y1), 3, color_kpt, -1)
                        for conn in mediapipe_connections:
                            pt1 = kpts[conn[0]]
                            pt2 = kpts[conn[1]]
                            cv2.line(frame, (int(pt1[0])+roi_x1, int(pt1[1])+roi_y1), (int(pt2[0])+roi_x1, int(pt2[1])+roi_y1), color_line, 2)
                    break
            if palm_in_this_frame and new_palm_roi is not None:
                last_palm_roi = new_palm_roi
                palm_frame_count += 1
            active_buffer.append(frame.copy())
            active_frame_counter += 1
            global_frame_counter += 1
            print(f"[{name}] [active] 已处理帧数: {global_frame_counter}")
            # 满30帧立即报警，否则采满75帧后再判断
            if palm_frame_count >= 30 or active_frame_counter >= alarm_buf_len:
                if palm_frame_count >= 30:
                    ts = time.strftime('%Y%m%d_%H%M%S')
                    alarm_path = os.path.join(alarm_dir, f'{name}_{ts}.mp4')
                    alarm_writer = cv2.VideoWriter(alarm_path, fourcc, fps if fps > 0 else 25, (width, height))
                    for f in active_buffer:
                        alarm_writer.write(f)
                    alarm_writer.release()
                    print(f"[{name}] !!! 触发报警：3秒内palm帧数{palm_frame_count}>=30，报警片段: {alarm_path}")
                    play_video_window(alarm_path, window_name=f'ALARM-{name}', wait=30)
                    state = 'cooldown'
                    cooldown_until = time.time() + cooldown_seconds
                else:
                    print(f"[{name}] 3秒内palm帧数{palm_frame_count}<30，丢弃片段，回到idle")
                    state = 'idle'
                continue

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
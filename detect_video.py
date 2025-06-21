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
    names = config.get('names', ['wave', 'nowave'])
    device = config.get('device', 'cpu')
    alarm_dir = config.get('alarm_dir', 'alarms')
    alarm_classes = config.get('alarm_classes', [0, 1])
    alarm_duration = int(config.get('alarm_duration', 3))
    detect_interval_ms = int(config.get('detect_interval_ms', 500))
    cooldown_seconds = int(config.get('cooldown_seconds', 60))
    skip_frame_on_error = config.get('skip_frame_on_error', True)
    error_recovery_delay = config.get('error_recovery_delay', 1)
    max_error_count = config.get('max_error_count', 3)

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
    
    # 显示视频解码配置
    video_decode_config = config.get('video_decode', {})
    if video_decode_config.get('use_hardware_decode', False):
        print(f"[{name}] 硬件解码已启用，后端: {video_decode_config.get('decode_backend', 'ffmpeg')}")
    else:
        print(f"[{name}] 使用软件解码")
    
    error_count = 0
    last_successful_frame_time = time.time()
    
    # 自动重连配置
    auto_reconnect = video_decode_config.get('auto_reconnect', True)
    reconnect_interval = video_decode_config.get('reconnect_interval', 120)
    max_decode_errors = video_decode_config.get('max_decode_errors', 50)
    error_check_window = video_decode_config.get('error_check_window', 30)
    
    last_reconnect_time = time.time()
    decode_error_count = 0
    last_error_reset_time = time.time()
    last_state_log_time = time.time()  # 上次状态日志时间
    
    if auto_reconnect:
        print(f"[{name}] 自动重连已启用，间隔: {reconnect_interval}秒")
    
    print(f"[{name}] 检测间隔设置: {detect_interval_ms}ms")
    print(f"[{name}] 初始状态: {state}")
    
    cap = None  # 初始化cap变量
    force_reconnect = False  # 强制重连标志
    
    # 合并为单一循环，同时处理重连和检测逻辑
    while True:
        # 记录循环开始时间（用于计算总检测周期）
        cycle_start_time = time.time()
        
        # 1. 状态日志 - 每30秒输出一次当前状态
        if (cycle_start_time - last_state_log_time) >= 30:
            print(f"[{name}] 当前状态: {state}, 解码错误计数: {decode_error_count}")
            last_state_log_time = cycle_start_time
        
        # 2. 检查是否需要定期重连
        if auto_reconnect and (cycle_start_time - last_reconnect_time) >= reconnect_interval:
            print(f"[{name}] 定期重连：已运行{reconnect_interval}秒，主动重新连接")
            force_reconnect = True
            last_reconnect_time = cycle_start_time
            decode_error_count = 0
            last_error_reset_time = cycle_start_time
        
        # 3. 检查是否因错误过多需要重连
        if auto_reconnect and (cycle_start_time - last_error_reset_time) >= error_check_window:
            if decode_error_count >= max_decode_errors:
                print(f"[{name}] 错误重连：{error_check_window}秒内发生{decode_error_count}个解码错误，强制重连")
                force_reconnect = True
                last_reconnect_time = cycle_start_time
            decode_error_count = 0
            last_error_reset_time = cycle_start_time
        
        # 4. 如果需要重连，先释放现有连接
        if force_reconnect and cap:
            print(f"[{name}] 执行重连，释放当前连接")
            cap.release()
            cap = None
            force_reconnect = False
        
        # 5. 冷却状态处理
        if state == 'cooldown':
            if cycle_start_time < cooldown_until:
                continue  # 继续冷却，但仍然检查重连
            else:
                print(f"[{name}] 状态切换: cooldown -> idle (冷却结束)")
                state = 'idle'
        
        # 6. 如果没有连接或连接无效，创建新连接
        if not cap or not cap.isOpened():
            print(f"[{name}] 创建新的视频连接...")
            cap = create_video_capture_with_hwdecode(url, config, name)
            if not cap or not cap.isOpened():
                print(f"[{name}] 无法打开视频流: {url}，5秒后重试")
                if cap:
                    cap.release()
                cap = None
                time.sleep(5)
                continue
            
            # 初始化缓冲区（每次重连都重新初始化）
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            alarm_buf_len = int(alarm_duration * (fps if fps > 0 else 25))
            frame_buffer = deque(maxlen=alarm_buf_len)
            alarm_flags = deque(maxlen=alarm_buf_len)
            wave_counts = deque(maxlen=alarm_buf_len)
            nowave_counts = deque(maxlen=alarm_buf_len)
            print(f"[{name}] 连接成功，fps={fps}, 分辨率={width}x{height}, 缓冲区长度={alarm_buf_len}")
        
        # 7. 读取帧
        ret, frame = cap.read()
        if not ret:
            print(f"[{name}] 读取帧失败，标记需要重连")
            decode_error_count += 1  # 增加解码错误计数
            if cap:
                cap.release()
            cap = None  # 重置cap变量
            continue  # 回到循环开始，会重新创建连接
        
        # 8. 检查帧质量
        if skip_frame_on_error and frame is not None:
            # 简单的帧质量检查：检查是否有明显的损坏像素
            frame_mean = frame.mean()
            if frame_mean < 1 or frame_mean > 254:  # 异常亮度值
                error_count += 1
                decode_error_count += 1  # 增加解码错误计数
                if error_count >= max_error_count:
                    print(f"[{name}] 检测到连续{error_count}个错误帧，跳过处理")
                    time.sleep(error_recovery_delay)
                    error_count = 0
                    continue
            else:
                error_count = 0
                last_successful_frame_time = time.time()
        
        # 9. 执行检测
        detect_start = time.time()
        results = model(frame, conf=confidence, verbose=False)[0]
        detect_end = time.time()
        detect_time_ms = (detect_end - detect_start) * 1000
        
        # 10. 处理检测结果
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
        
        # 11. 状态切换逻辑（添加详细日志）
        old_state = state
        if state == 'idle':
            if has_alarm_class:
                state = 'active'
                print(f"[{name}] 状态切换: idle -> active (检测到 wave:{wave_count}, nowave:{nowave_count})")
        elif state == 'active':
            if not has_alarm_class:
                state = 'idle'
                print(f"[{name}] 状态切换: active -> idle (未检测到目标)")
        
        # 12. 更新缓冲区（只在非冷却状态下且缓冲区已初始化）
        if (state != 'cooldown' and frame_buffer is not None and 
            alarm_flags is not None and wave_counts is not None and nowave_counts is not None):
            frame_buffer.append(frame.copy())
            alarm_flags.append(has_alarm_class)
            wave_counts.append(wave_count)
            nowave_counts.append(nowave_count)
        
        # 13. 报警逻辑（只在active状态下检查且缓冲区已初始化）
        if (state == 'active' and alarm_flags is not None and wave_counts is not None and 
            nowave_counts is not None and len(alarm_flags) == alarm_buf_len):
            if all(alarm_flags):
                total_wave = sum(wave_counts)
                total_nowave = sum(nowave_counts)
                if total_nowave == 0:
                    ratio = float('inf') if total_wave > 0 else 0
                else:
                    ratio = total_wave / total_nowave
                
                print(f"[{name}] 报警检查: 连续{alarm_duration}秒检测到目标, wave:{total_wave}, nowave:{total_nowave}, 比例:{ratio:.2f}")
                
                if ratio >= 3:
                    ts = time.strftime('%Y%m%d_%H%M%S')
                    alarm_path = os.path.join(alarm_dir, f'{name}_{ts}.mp4')
                    alarm_writer = cv2.VideoWriter(alarm_path, fourcc, fps if fps > 0 else 25, (width, height))
                    if frame_buffer is not None:
                        for f in frame_buffer:
                            alarm_writer.write(f)
                    alarm_writer.release()
                    print(f"[{name}] !!! 触发报警：wave/nowave比例{ratio:.2f}>=3，报警片段: {alarm_path}")
                    play_video_window(alarm_path, window_name=f'ALARM-{name}', wait=int(1000/(fps if fps > 0 else 25)))
                    state = 'cooldown'
                    cooldown_until = time.time() + cooldown_seconds
                    print(f"[{name}] 状态切换: active -> cooldown (报警触发，冷却{cooldown_seconds}秒)")
        
        # 14. 检测间隔控制 - 根据实际检测耗时动态调整sleep时间
        cycle_end_time = time.time()
        total_cycle_time_ms = (cycle_end_time - cycle_start_time) * 1000
        
        # 计算需要sleep的时间
        sleep_time_ms = detect_interval_ms - total_cycle_time_ms
        
        if sleep_time_ms > 0:
            sleep_time_seconds = sleep_time_ms / 1000.0
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{name}] 检测耗时: {detect_time_ms:.2f}ms, 总周期: {total_cycle_time_ms:.2f}ms, sleep: {sleep_time_ms:.2f}ms, 状态: {state}")
            time.sleep(sleep_time_seconds)
        else:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{name}] 检测耗时: {detect_time_ms:.2f}ms, 总周期: {total_cycle_time_ms:.2f}ms, 无需sleep (超时), 状态: {state}")
    
    # 清理资源
    if cap:
        cap.release()
        print(f"[{name}] 进程结束，释放资源")

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
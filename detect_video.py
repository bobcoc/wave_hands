import cv2
import yaml
import os
import sys
import time
from collections import deque
from multiprocessing import Process
import multiprocessing
from alarm_video_popup import show_alarm_video_popup
import requests
from urllib.parse import urlparse, parse_qs
from hand_detector import HandDetector

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

def upload_file_to_wechat(file_path, webhook_url):
    """上传文件到企业微信群，返回media_id"""
    try:
        parsed = urlparse(webhook_url)
        key = parse_qs(parsed.query).get('key', [None])[0]
        if not key:
            print("[WeChat] webhook_url缺少key参数")
            return None
        upload_url = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/upload_media?key={key}&type=file"
        with open(file_path, 'rb') as f:
            files = {'media': (os.path.basename(file_path), f, 'application/octet-stream')}
            resp = requests.post(upload_url, files=files)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('errcode') == 0 and 'media_id' in data:
                return data['media_id']
            else:
                print(f"[WeChat] 上传文件失败: {data}")
        else:
            print(f"[WeChat] 上传文件HTTP错误: {resp.status_code}")
    except Exception as e:
        print(f"[WeChat] 上传文件异常: {e}")
    return None

def send_wechat_file_message(webhook_url, media_id):
    """通过Webhook推送文件类型消息到群"""
    try:
        payload = {
            "msgtype": "file",
            "file": {"media_id": media_id}
        }
        resp = requests.post(webhook_url, json=payload)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('errcode') == 0:
                print("[WeChat] 文件消息推送成功")
            else:
                print(f"[WeChat] 文件消息推送失败: {data}")
        else:
            print(f"[WeChat] 文件消息推送HTTP错误: {resp.status_code}")
    except Exception as e:
        print(f"[WeChat] 文件消息推送异常: {e}")

def send_wechat_text_message(webhook_url, text):
    """通过Webhook推送文字消息到群"""
    try:
        payload = {
            "msgtype": "text",
            "text": {"content": text}
        }
        resp = requests.post(webhook_url, json=payload)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('errcode') == 0:
                print("[WeChat] 文字消息推送成功")
            else:
                print(f"[WeChat] 文字消息推送失败: {data}")
        else:
            print(f"[WeChat] 文字消息推送HTTP错误: {resp.status_code}")
    except Exception as e:
        print(f"[WeChat] 文字消息推送异常: {e}")

def worker(stream_cfg, config):
    name = stream_cfg.get('name', 'noname')
    url = stream_cfg.get('url')
    weights = config.get('weights', 'weight/best.pt')
    confidence = float(config.get('confidence', 0.2))
    device = config.get('device', 'cpu')
    alarm_dir = config.get('alarm_dir', 'alarms')
    alarm_duration = int(config.get('alarm_duration', 3))
    cooldown_seconds = int(config.get('cooldown_seconds', 60))
    idle_detect_interval = int(config.get('idle_detect_interval', 5))  # idle下每N帧检测一次
    alarm_video_overlay_level = int(config.get('alarm_video_overlay_level', 2))
    font_scale = float(config.get('font_scale', 0.4))
    font_thickness = int(config.get('font_thickness', 1))
    alarm_frame_threshold = int(config.get('alarm_frame_threshold', 10))  # 触发报警的帧数阈值

    if not os.path.exists(alarm_dir):
        os.makedirs(alarm_dir)

    # 初始化检测器
    detector = HandDetector(
        detector='mediapipe',  # 使用YOLO检测手掌位置，MediaPipe提取关键点
        weights=weights,
        confidence=confidence,
        device=device,
        font_scale=font_scale,
        font_thickness=font_thickness
    )

    state = 'idle'  # idle, active, cooldown
    cooldown_until = 0
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    print(f"[{name}] 进程启动，流地址: {url}")

    cap = None
    idle_frame_counter = 0
    alarm_buf_len = 75  # 3秒*25帧，后续可根据fps动态调整
    global_frame_counter = 0

    while True:
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

        if state == 'idle':
            # 跳帧检测
            for _ in range(idle_detect_interval-1):
                cap.grab()
                time.sleep(0.02)
            ret, frame = cap.read()
            if not ret:
                print(f"[{name}] idle模式读取帧失败，重连")
                cap.release()
                cap = None
                continue

            # 处理当前帧
            output, hands_info = detector.process_frame(frame)
            # 只有检测到手掌（Palm）才算
            palm_found = any(hand['is_palm_up'] for hand in hands_info)
            
            global_frame_counter += 1
            print(f"[{name}] [idle] 已处理帧数: {global_frame_counter}")

            if palm_found:
                print(f"[{name}] 状态切换: idle -> active (检测到palm)")
                state = 'active'
                active_buffer = [frame.copy()]
                palm_frame_count = 1
                active_frame_counter = 1
                continue

        elif state == 'active':
            ret, frame = cap.read()
            if not ret:
                print(f"[{name}] active模式读取帧失败，重连")
                cap.release()
                cap = None
                state = 'idle'
                continue

            # 处理当前帧
            output, hands_info = detector.process_frame(frame)
            # 只有检测到手掌（Palm）才算
            palm_in_this_frame = any(hand['is_palm_up'] for hand in hands_info)
            
            if palm_in_this_frame:
                palm_frame_count += 1

            active_buffer.append(frame.copy())
            active_frame_counter += 1
            global_frame_counter += 1
            print(f"[{name}] [active] 已处理帧数: {global_frame_counter}, palm帧数: {palm_frame_count}/{active_frame_counter}")

            # 满alarm_frame_threshold帧立即报警，否则采满75帧后再判断
            if active_frame_counter >= alarm_buf_len:
                if palm_frame_count >= alarm_frame_threshold:
                    ts = time.strftime('%Y%m%d_%H%M%S')
                    alarm_path = os.path.join(alarm_dir, f'{name}_{ts}.mp4')
                    alarm_writer = cv2.VideoWriter(alarm_path, fourcc, fps if fps > 0 else 25, (width, height))
                    
                    for idx, f in enumerate(active_buffer):
                        if alarm_video_overlay_level > 0:
                            # 使用detector处理每一帧
                            processed_frame, _ = detector.process_frame(f)
                            alarm_writer.write(processed_frame)
                        else:
                            alarm_writer.write(f)
                    
                    alarm_writer.release()
                    print(f"[{name}] !!! 触发报警：3秒内palm帧数{palm_frame_count}>={alarm_frame_threshold}，报警片段: {alarm_path}")
                    
                    # 弹出报警窗口
                    multiprocessing.Process(target=show_alarm_video_popup, 
                                         args=(alarm_path, f'ALARM-{name}')).start()
                    
                    # 微信报警推送
                    try:
                        webhook_url = config.get('wechat_webhook_url')
                        if webhook_url:
                            text_msg = f"{name}有老师举手，请及时处理"
                            send_wechat_text_message(webhook_url, text_msg)
                            media_id = upload_file_to_wechat(alarm_path, webhook_url)
                            if media_id:
                                send_wechat_file_message(webhook_url, media_id)
                            else:
                                print(f"[WeChat] 未获取到media_id，文件推送失败")
                        else:
                            print("[WeChat] 未配置wechat_webhook_url，跳过微信推送")
                    except Exception as e:
                        print(f"[WeChat] 微信报警推送异常: {e}")
                    
                    state = 'cooldown'
                    cooldown_until = time.time() + cooldown_seconds
                else:
                    print(f"[{name}] 3秒内palm帧数{palm_frame_count}<{alarm_frame_threshold}，丢弃片段，回到idle")
                    state = 'idle'
                
                # 重置状态
                active_buffer = []
                palm_frame_count = 0
                active_frame_counter = 0
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
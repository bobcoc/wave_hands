import cv2
import yaml
import os
import sys
import time
from multiprocessing import Process
import multiprocessing
from alarm_video_popup import show_alarm_video_popup
import requests
from urllib.parse import urlparse, parse_qs
from hand_detector import HandDetector

# 抑制MediaPipe和FFmpeg的警告日志
os.environ['GLOG_minloglevel'] = '2'  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 抑制TensorFlow日志
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'loglevel;quiet'  # 抑制FFmpeg解码错误

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
                # 设置缓冲区大小（注意：对RTSP流通常无效，需要主动清空缓冲区）
                cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
                # 尝试设置硬件解码相关参数
                try:
                    # 设置解码器优化参数
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('H', '2', '6', '4'))
                    # 设置更多容错参数
                    cap.set(cv2.CAP_PROP_FPS, 25)  # 限制FPS减少缓冲压力
                    # 设置RTSP传输协议为TCP（更稳定但延迟更高）
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30000)  # 30秒超时
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000)  # 1秒读取超时
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
    idle_detect_interval = float(config.get('idle_detect_interval', 3.0))  # idle下每N秒检测一次
    alarm_video_overlay_level = int(config.get('alarm_video_overlay_level', 2))
    font_scale = float(config.get('font_scale', 0.4))
    font_thickness = int(config.get('font_thickness', 1))
    alarm_frame_threshold = int(config.get('alarm_frame_threshold', 10))  # 触发报警的帧数阈值
    enable_alarm_popup = config.get('enable_alarm_popup', False)  # 是否启用报警弹窗，默认关闭

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
    print(f"[{name}] 进程启动,流地址: {url}")

    cap = None
    alarm_buf_len = 75  # 3秒*25帧，后续可根据fps动态调整
    global_frame_counter = 0
    print_interval = 10  # 每N帧打印一次，减少I/O开销
    last_frame_time = time.time()  # 用于计算帧间隔
    last_idle_detect_time = time.time()  # 上次idle检测的时间戳（初始化为当前时间）
    ccc = 0
    while True:
        if state == 'cooldown':
            if time.time() < cooldown_until:
                # 持续清空缓冲区，避免积压旧帧
                ret = cap.grab()
                continue
            else:
                print(f"[{name}] 状态切换: cooldown -> idle (冷却结束)")
                state = 'idle'
                # 不需要额外清空，cooldown期间已持续清空了
                print(f"[{name}] 缓冲区已清空，从实时帧开始检测")

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
            current_time = time.time()
            
            # 检查是否到达检测间隔时间
            time_since_last = current_time - last_idle_detect_time
            if time_since_last < idle_detect_interval:
                # 预测性避免grab卡住：估算理论上应该积累的帧数
                if fps > 0:
                    # 理论帧数 = 已经过去的时间 * 帧率
                    expected_frames = int(time_since_last * fps)
                    # 保留一些安全边际，当grab次数接近理论帧数时停止
                    safety_margin = 5  # 保留5帧的安全边际
                    if ccc >= expected_frames - safety_margin:
                        # 已经接近缓冲区底部，休眠等待新帧
                        time.sleep(0.025)  # 休眠40ms，约1帧时间
                # 正常grab清空缓冲区
                ret = cap.grab()
                if not ret:
                    # grab失败说明流有问题，直接重连
                    print(f"[{name}] grab失败，流可能中断，重连")
                    cap.release()
                    cap = None
                    ccc = 0
                    continue
                ccc = ccc + 1
                continue
            # print(ccc)
            grab_count = ccc  # 记录grab次数用于调试
            ccc = 0
            # 计算距离上次检测的实际时间间隔（在更新时间戳之前计算）
            actual_interval = current_time - last_idle_detect_time
            
            # 读取当前最新帧
            read_start = time.time()
            ret, frame = cap.read()
            read_time = time.time() - read_start
            
            if not ret:
                print(f"[{name}] idle模式读取帧失败，重连")
                cap.release()
                cap = None
                continue
            
            # 处理当前帧
            process_start = time.time()
            try:
                output, hands_info = detector.process_frame(frame)
            except Exception as e:
                print(f"[{name}] [ERROR] 检测器处理帧时出错: {e}")
                import traceback
                traceback.print_exc()
                # 出错时跳过这一帧，继续下一次检测
                continue
            process_time = time.time() - process_start
            
            # 在处理完成后再更新时间戳，确保包含读取和处理的耗时
            last_idle_detect_time = time.time()
            last_frame_time = last_idle_detect_time  # 更新帧时间（供active状态使用）
                
            # 只有检测到手掌（Palm）才算
            palm_found = any(hand['is_palm_up'] for hand in hands_info)
            
            global_frame_counter += 1
            
            # 在第1250次检测时保存一张图片，用于验证实时性
            if False and global_frame_counter == 1250:
                try:
                    debug_image_dir = os.path.join(alarm_dir, 'realtime_check')
                    if not os.path.exists(debug_image_dir):
                        os.makedirs(debug_image_dir)
                    # 使用当前系统时间作为文件名
                    current_timestamp = time.strftime('%Y%m%d_%H%M%S')
                    debug_image_path = os.path.join(debug_image_dir, f'{name}_realtime_check_{current_timestamp}.jpg')
                    cv2.imwrite(debug_image_path, frame)
                    print(f"[{name}] [实时性验证] 已保存第150次检测的图片: {debug_image_path}")
                    print(f"[{name}] [实时性验证] 请对比图片中监控时间与文件名时间是否一致")
                except Exception as e:
                    print(f"[{name}] [ERROR] 保存实时性验证图片失败: {e}")

            # 每次检测都保存一张图片（用于调试查看是哪个教室）
            '''
            print(f"[{name}] [DEBUG] global_frame_counter={global_frame_counter}")
            if global_frame_counter <= 55:  # 只保存前55次检测的图片
                try:
                    debug_image_dir = os.path.join(alarm_dir, 'debug_frames')
                    print(f"[{name}] [DEBUG] debug_image_dir={debug_image_dir}")
                    if not os.path.exists(debug_image_dir):
                        os.makedirs(debug_image_dir)
                        print(f"[{name}] [DEBUG] 创建目录: {debug_image_dir}")
                    debug_image_path = os.path.join(debug_image_dir, f'{name}_frame_{global_frame_counter}.jpg')
                    print(f"[{name}] [DEBUG] 即将保存图片到: {debug_image_path}")
                    cv2.imwrite(debug_image_path, frame)
                    print(f"[{name}] [DEBUG] 已保存调试图片: {debug_image_path}")
                except Exception as e:
                    print(f"[{name}] [ERROR] 保存调试图片失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[{name}] [DEBUG] 跳过保存，global_frame_counter({global_frame_counter}) > 55")
            '''
            # 每次检测都打印信息（因为检测频率已经降低）
            print(f"[{name}] [idle] 检测#{global_frame_counter}, 检测间隔: {actual_interval:.1f}s, 检测到手: {len(hands_info)}个, Palm: {palm_found}, 读取耗时: {read_time*1000:.1f}ms, 处理耗时: {process_time*1000:.1f}ms, grab次数: {grab_count}")
            
            # 调试：检测到任何手时都打印详细信息
            if len(hands_info) > 0:
                for i, hand in enumerate(hands_info):
                    print(f"[{name}] [DEBUG] 手#{i+1}: {hand['handedness']}, is_palm_up={hand['is_palm_up']}, conf={hand.get('confidence', 0):.2f}")
            if palm_found:
                print(f"[{name}] 状态切换: idle -> active (检测到palm)")
                state = 'active'
                # 需要复制帧，因为要保存到缓冲区
                active_buffer = [frame.copy()]
                active_buffer_processed = [output.copy()]  # 保存已处理的帧
                palm_frame_count = 1
                active_frame_counter = 1
                continue

        elif state == 'active':
            # 记录读取帧的时间
            read_start = time.time()
            ret, frame = cap.read()
            read_time = time.time() - read_start
            
            if not ret:
                print(f"[{name}] active模式读取帧失败，重连")
                cap.release()
                cap = None
                state = 'idle'
                continue
            
            # 计算帧间隔和估算缓冲区积压
            current_time = time.time()
            frame_interval = current_time - last_frame_time
            last_frame_time = current_time
            
            expected_interval = 1.0 / fps if fps > 0 else 0.04
            if expected_interval > 0:
                estimated_buffer_frames = max(0, int((read_time - expected_interval) * fps)) if fps > 0 else 0
            else:
                estimated_buffer_frames = 0

            # 处理当前帧
            output, hands_info = detector.process_frame(frame)
            # 只有检测到手掌（Palm）才算
            palm_in_this_frame = any(hand['is_palm_up'] for hand in hands_info)
            
            # 调试：打印每帧的棅测结果
            if len(hands_info) > 0:
                for i, hand in enumerate(hands_info):
                    print(f"[{name}] [active-DEBUG] 手#{i+1}: {hand['handedness']}, is_palm_up={hand['is_palm_up']}, conf={hand.get('confidence', 0):.2f}")
            
            if palm_in_this_frame:
                palm_frame_count += 1

            # 必须复制帧，避免opencv缓冲区复用导致数据覆盖
            active_buffer.append(frame.copy())
            active_buffer_processed.append(output.copy())  # 保存已处理的帧
            active_frame_counter += 1
            global_frame_counter += 1
            # 优化：减少打印频率
            if active_frame_counter % print_interval == 0 or palm_in_this_frame:
                buffer_info = f", 缓冲: ~{estimated_buffer_frames}帧" if estimated_buffer_frames > 5 else ""
                print(f"[{name}] [active] 已处理: {global_frame_counter}, palm: {palm_frame_count}/{active_frame_counter}, 帧间隔: {frame_interval*1000:.1f}ms{buffer_info}")

            # 满alarm_frame_threshold帧立即报警，否则采满75帧后再判断
            if active_frame_counter >= alarm_buf_len:
                if palm_frame_count >= alarm_frame_threshold:
                    ts = time.strftime('%Y%m%d_%H%M%S')
                    alarm_path = os.path.join(alarm_dir, f'{name}_{ts}.mp4')
                    alarm_writer = cv2.VideoWriter(alarm_path, fourcc, fps if fps > 0 else 25, (width, height))
                    
                    # 优化：复用已处理的帧，避免重复检测
                    for idx in range(len(active_buffer)):
                        if alarm_video_overlay_level > 0:
                            # 使用已处理过的帧
                            alarm_writer.write(active_buffer_processed[idx])
                        else:
                            # 使用原始帧
                            alarm_writer.write(active_buffer[idx])
                    
                    alarm_writer.release()
                    print(f"[{name}] !!! 触发报警：3秒内palm帧数{palm_frame_count}>={alarm_frame_threshold}，报警片段: {alarm_path}")
                    
                    # 弹出报警窗口（可配置）
                    if enable_alarm_popup:
                        alarm_popup_process = multiprocessing.Process(
                            target=show_alarm_video_popup, 
                            args=(alarm_path, f'ALARM-{name}')
                        )
                        alarm_popup_process.start()
                        print(f"[{name}] 已启动报警弹窗")
                    else:
                        print(f"[{name}] 报警弹窗已禁用（enable_alarm_popup=False）")
                    
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
                    
                    # 清空视频流缓冲区，避免处理积压的旧帧
                    print(f"[{name}] 清空视频缓冲区...")
                    for _ in range(50):  # 清空最多50帧的积压
                        cap.grab()
                    print(f"[{name}] 缓冲区已清空")
                
                # 重置状态
                active_buffer = []
                active_buffer_processed = []
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
        # 不设置为守护进程，通过 terminate/kill 手动清理
        p.start()
        processes.append(p)
        print(f"[主进程] 启动进程处理: {stream_cfg.get('name')} (PID: {p.pid})")
    
    print(f"[主进程] 所有进程已启动，按Ctrl+C退出")
    
    try:
        # 等待所有进程
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n[主进程] 接收到键盘中断 (Ctrl+C)，正在停止所有进程...")
        # 终止所有子进程
        for p in processes:
            if p.is_alive():
                print(f"[主进程] 终止进程 PID: {p.pid}")
                p.terminate()
        
        # 等待所有进程真正退出（最多等3秒）
        for p in processes:
            p.join(timeout=3)
            if p.is_alive():
                print(f"[主进程] 强制杀死进程 PID: {p.pid}")
                p.kill()  # 强制杀死仍未退出的进程
    
    print("[主进程] 程序正常退出")

if __name__ == '__main__':
    main() 
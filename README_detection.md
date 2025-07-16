# 手掌检测程序使用说明

本项目包含两个独立的手掌检测程序，都使用相同的核心检测代码 `hand_detection_core.py`。

## 程序说明

### 1. detect_video.py - 视频流检测程序
专门用于处理实时视频流（如RTSP、HTTP流等），支持多进程并发处理多个视频流。

**特点：**
- 支持硬件解码加速
- 多进程并发处理多个视频流
- 自动重连机制
- 实时报警功能
- 支持企业微信推送

### 2. detect_local_video.py - 本地视频检测程序
专门用于处理本地视频文件，逐帧检测手掌。

**特点：**
- 逐帧检测，不跳帧
- 检测到手掌立即进入active状态
- 适合离线视频分析
- 输出报警视频片段

## 配置文件 (config.yaml)

两个程序共用同一个配置文件，但使用不同的配置项：

```yaml
# 通用配置
weights: 'weight/best.pt'  # 模型权重文件
confidence: 0.5           # 检测置信度阈值
device: 'cpu'             # 设备类型 (cpu/cuda)
alarm_dir: 'alarms'       # 报警视频保存目录
alarm_duration: 3         # 报警视频时长(秒)
alarm_video_overlay_level: 0  # 视频叠加级别 (0=无, 1=框, 2=框+关键点)

# 视频流检测配置
streams:
  - name: 'camera1'
    url: 'rtsp://192.168.1.100:554/stream1'
  - name: 'camera2'
    url: 'rtsp://192.168.1.101:554/stream1'

# 本地视频检测配置
video_file: 'path/to/your/video.mp4'

# 视频解码配置
video_decode:
  use_hardware_decode: false
  buffer_size: 1
  decode_backend: 'ffmpeg'
  fallback_to_software: true
  rtsp_transport: 'tcp'

# 企业微信配置（可选）
wechat:
  webhook_url: 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=YOUR_KEY'
```

## 使用方法

### 视频流检测
```bash
python detect_video.py
```

程序会自动读取 `config.yaml` 中的 `streams` 配置，为每个视频流创建独立的进程进行检测。

### 本地视频检测
```bash
python detect_local_video.py
```

程序会读取 `config.yaml` 中的 `video_file` 配置，对指定的本地视频文件进行检测。

## 核心检测模块

两个程序都使用 `hand_detection_core.py` 作为核心检测模块，包含：

- `load_yolo_model()`: 加载YOLO模型
- `detect_hands()`: 手部检测
- `classify_hand_pose()`: 手势分类
- `draw_hand_overlay()`: 绘制检测结果

## 检测逻辑

### 状态机
两个程序都使用相同的状态机逻辑：

1. **idle状态**: 检测手掌
   - 视频流：每N帧检测一次（可配置）
   - 本地视频：每帧都检测

2. **active状态**: 检测到手掌后进入
   - 在ROI区域内继续检测
   - 记录视频片段
   - 统计palm帧数

3. **报警条件**:
   - palm帧数 >= 30 或
   - active帧数 >= alarm_duration * fps

### 报警输出
满足报警条件时，会输出一个MP4视频片段到 `alarm_dir` 目录，文件名格式：
- 视频流：`{stream_name}_{timestamp}.mp4`
- 本地视频：`local_video_{timestamp}.mp4`

## 依赖项

```bash
pip install opencv-python ultralytics pyyaml numpy requests
```

## 注意事项

1. 确保模型权重文件 `weight/best.pt` 存在
2. 视频流检测需要稳定的网络连接
3. 本地视频检测会逐帧处理，大文件可能需要较长时间
4. 报警视频会占用磁盘空间，注意定期清理
5. 企业微信推送需要配置正确的webhook URL

## 故障排除

### 视频流连接失败
- 检查网络连接
- 验证RTSP URL格式
- 尝试不同的传输协议（TCP/UDP）

### 检测效果不佳
- 调整 `confidence` 阈值
- 检查模型权重文件
- 确认视频质量

### 性能问题
- 启用硬件解码（需要支持GPU）
- 调整 `idle_detect_interval` 参数
- 使用GPU加速（设置 `device: 'cuda'`） 
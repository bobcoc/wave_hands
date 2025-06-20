# 多路视频流智能检测工具（detect_video.py）

本工具支持同时监控多路RTSP视频流，基于YOLO模型自动检测"挥手/不挥手"目标，实现高效报警与资源优化。适用于考场监控、安防等场景。

## 功能简介
- 支持多路RTSP/本地视频流并发检测（推荐50路及以上，视硬件性能调整）
- 检测"wave/nowave"目标，报警逻辑灵活可配
- 空闲时可配置检测间隔，节省算力
- 报警后自动冷却，避免重复报警
- 每路流独立进程，互不影响
- 报警片段自动保存，命名清晰
- 详细日志输出，便于追踪与排查

## 配置文件示例（config.yaml）
```yaml
weights: weight/best.pt
confidence: 0.7
names: ['wave', 'nowave']
device: cpu
alarm_dir: alarms
alarm_classes: [0, 1]  # 0: wave, 1: nowave
alarm_duration: 3  # 连续检测多少秒触发报警
# 新增参数：
detect_interval_ms: 500  # 空闲时检测间隔（毫秒）
cooldown_seconds: 60     # 报警后冷却时长（秒）
streams:
  - name: A202
    url: rtsp://admin:HuaWei@12345@10.0.14.78:554/LiveMedia/ch1/Media1
  - name: A203
    url: rtsp://admin:HuaWei@12345@10.0.14.79:554/LiveMedia/ch1/Media1
    ......
```

## 参数说明
- `streams`：多路流配置，每个流包含`name`和`url`
- `detect_interval_ms`：空闲状态下每隔多少毫秒检测一帧
- `cooldown_seconds`：报警后该流冷却多少秒，这个期间不再检测，既节省服务器资源，又可以避免重复报警
- 其它参数同YOLO模型推理相关

## 使用方法
1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 配置`config.yaml`，填写所有视频流地址及参数
3. 运行检测：
   ```bash
   python detect_video.py
   ```

## 报警与输出
- 满足报警条件（如连续N秒wave:nowave≥3）时，自动保存报警片段至`alarm_dir`，命名为`{流名}_{时间戳}.mp4`
- 报警后该流冷却`cooldown_seconds`秒，期间不再检测
- 日志输出每路流的状态变迁、报警、冷却等详细信息

## 依赖环境
- Python >= 3.7
- OpenCV >= 4.5.0
- Ultralytics >= 8.0.0
- PyYAML >= 6.0

## 许可证
MIT License 
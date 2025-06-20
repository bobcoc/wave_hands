# 

这个Python程序可以从监考视频中提取左下角四分之一画面，每200ms截取一帧，使用YOLO模型检测人物，并生成标准的YOLO格式数据集。

## 功能特点

- 🎥 **视频处理**: 自动处理视频文件，提取指定区域
- ✂️ **智能裁剪**: 提取左下角四分之一画面
- ⏱️ **定时截取**: 可配置的时间间隔截取帧（默认200ms）
- 👥 **人物检测**: 使用YOLO模型自动检测人物
- 📊 **数据集生成**: 输出标准YOLO格式的数据集
- 📈 **可视化**: 可选的检测结果可视化功能

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python video_to_yolo_dataset.py 你的视频文件.mp4
```

### 高级用法

```bash
python video_to_yolo_dataset.py 你的视频文件.mp4 \
    --output_dir my_dataset \
    --interval 500 \
    --confidence 0.6 \
    --visualize
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `video_path` | str | 必需 | 输入视频文件路径 |
| `--output_dir` | str | `yolo_dataset` | 输出数据集目录 |
| `--model` | str | `yolov8n.pt` | YOLO模型文件路径 |
| `--interval` | int | `200` | 帧提取间隔（毫秒） |
| `--confidence` | float | `0.5` | 检测置信度阈值 |
| `--visualize` | flag | `False` | 是否可视化检测结果 |

## 输出结构

程序运行后会生成以下目录结构：

```
yolo_dataset/
├── images/              # 图像文件
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   └── ...
├── labels/              # YOLO格式标注文件
│   ├── frame_000000.txt
│   ├── frame_000001.txt
│   └── ...
├── dataset.yaml         # 数据集配置文件
└── detection_samples.png # 检测结果可视化（可选）
```

## YOLO标注格式

每个`.txt`文件包含检测到的人物边界框信息，格式为：
```
class_id center_x center_y width height
```

其中：
- `class_id`: 类别ID（0表示人）
- `center_x, center_y`: 边界框中心点坐标（归一化到0-1）
- `width, height`: 边界框宽度和高度（归一化到0-1）

## 示例

### 处理监考视频

假设你有一个名为`exam_video.mp4`的监考视频：

```bash
# 基本处理，每200ms提取一帧
python video_to_yolo_dataset.py exam_video.mp4

# 每1秒提取一帧，置信度阈值0.7，并显示可视化结果
python video_to_yolo_dataset.py exam_video.mp4 \
    --interval 1000 \
    --confidence 0.7 \
    --visualize \
    --output_dir exam_dataset
```

### 在代码中使用

```python
from video_to_yolo_dataset import VideoToYOLODataset

# 创建转换器
converter = VideoToYOLODataset(
    video_path="your_video.mp4",
    output_dir="my_dataset",
    model_path="yolov8n.pt"
)

# 提取帧并检测
num_frames = converter.extract_frames_and_detect(
    interval_ms=300,  # 每300ms一帧
    confidence_threshold=0.6
)

# 可视化结果
converter.visualize_detections(num_samples=10)
```

## 技术说明

### 视频区域提取

程序会自动计算左下角四分之一区域：
- 裁剪起始点：(0, height/2)
- 裁剪尺寸：(width/2, height/2)

### YOLO模型

- 默认使用YOLOv8n模型（轻量级）
- 只检测人类（COCO数据集中的class 0）
- 支持自定义模型路径

### 性能优化

- 只在指定间隔处理帧，提高处理效率
- 使用置信度阈值过滤低质量检测
- 支持批量处理和并行计算

## 故障排除

### 常见问题

1. **视频无法打开**
   - 检查视频文件路径是否正确
   - 确保视频格式受OpenCV支持

2. **YOLO模型下载失败**
   - 首次运行会自动下载模型
   - 检查网络连接，或手动下载模型文件

3. **内存不足**
   - 增大处理间隔（--interval）
   - 使用更小的YOLO模型

4. **检测效果不佳**
   - 调整置信度阈值（--confidence）
   - 尝试使用更大的YOLO模型

## 依赖版本

- Python >= 3.7
- OpenCV >= 4.5.0
- Ultralytics >= 8.0.0
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- PyYAML >= 6.0

## 许可证

MIT License

# 多路视频流检测说明

## 配置文件config.yaml示例

```
weights: weight/best.pt
confidence: 0.7
names: ['wave', 'nowave']
device: cpu
alarm_dir: alarms
alarm_classes: [0, 1]  # 0: wave, 1: nowave
alarm_duration: 3  # 秒
detect_interval_ms: 500  # 检测间隔，单位ms
cooldown_seconds: 60  # 报警冷却时长，单位秒
streams:
  - name: cam1
    url: rtsp://example.com/stream1
  - name: cam2
    url: rtsp://example.com/stream2
```

- `streams`: 配置多路视频流，每个流包含name和url。
- `detect_interval_ms`: 空闲状态下每隔多少毫秒检测一帧（如500表示每500ms检测一次）。
- `cooldown_seconds`: 报警后该流冷却多少秒，冷却期间不再检测。
- 其它参数与单路检测一致。

## 运行方式

```
python detect_video.py
```

程序会自动为每个流启动独立进程，互不影响。

## 日志与报警

- 每路流的状态变迁、报警、冷却等会在控制台输出详细日志。
- 报警片段以`{流名}_{时间戳}.mp4`命名，保存在`alarm_dir`目录下。 
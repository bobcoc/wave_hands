# MediaPipe Holistic 挥手检测程序使用说明

## 🎯 功能概述

本程序使用MediaPipe Holistic技术替代原版的YOLO+DeepSORT方案，实现更精确的挥手检测。

## 📦 安装依赖

```bash
pip install -r requirements_mediapipe.txt
```

## 🔧 配置文件

使用与原版相同的 `config.yaml` 配置文件，支持所有原有参数：

```yaml
teacher_height_threshold: 400  # 教师身高过滤阈值（像素）
wave_change_threshold: 3       # 挥手变化阈值
confidence: 0.7               # 检测置信度
device: 'cpu'                 # 设备类型
input_video: 'input.mp4'      # 输入视频路径
alarm_dir: 'alarms'           # 报警片段保存目录
alarm_duration: 3             # 报警持续时间（秒）
```

## 🚀 运行程序

```bash
python mediapipe_wave_detector.py
```

## 🔍 核心改进

### 1. **检测精度提升**
- ✅ 解决YOLO边界框不完整问题
- ✅ 直接检测人体关键点，无边界框依赖
- ✅ 精确的手臂角度计算

### 2. **技术对比**

| 方面 | 原版 (YOLO+DeepSORT) | 新版 (MediaPipe Holistic) |
|------|---------------------|---------------------------|
| 人员检测 | YOLO边界框 | 33个人体关键点 |
| 手部检测 | 边界框尺寸变化 | 右手臂角度变化 |
| 追踪方式 | DeepSORT ID | 基于人体中心点 |
| 检测精度 | 可能遗漏伸展手臂 | 准确捕获所有手臂动作 |
| 免训练 | 需要YOLO权重 | MediaPipe预训练模型 |

### 3. **输出文件**

程序会生成以下输出文件：

- `mediapipe_wave_analysis/`: 处理后的视频文件
- `alarms/`: 检测到挥手的报警片段
- `*_arm_angles_*.txt`: 手臂角度数据日志

## 📊 数据格式

### 角度数据日志格式：
```
frame_id,person_id,right_arm_angle,height_pixels,center_x,center_y
1,1,145.23,450,0.512,0.345
2,1,142.18,452,0.515,0.347
...
```

## 💡 使用建议

1. **性能优化**：
   - 如果性能不足，可以降低视频分辨率
   - 调整MediaPipe的`model_complexity`参数

2. **检测调优**：
   - 调整`teacher_height_threshold`适应不同场景
   - 调整`wave_change_threshold`控制检测敏感度

3. **多人场景**：
   - 程序支持多人同时检测
   - 自动分配和追踪人员ID

## 🎥 实时显示

程序会实时显示：
- 人体姿态关键点（橙色）
- 右手关键点（绿色）
- 人员ID和状态信息
- 检测统计信息

按 'q' 键退出程序。

## 🔧 故障排除

1. **MediaPipe导入错误**：
   ```bash
   pip install mediapipe
   ```

2. **性能问题**：
   - 降低视频分辨率
   - 使用GPU加速（如果支持）

3. **检测精度不佳**：
   - 确保光照充足
   - 调整摄像头角度
   - 优化`teacher_height_threshold`参数 
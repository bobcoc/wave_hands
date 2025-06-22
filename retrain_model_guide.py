import cv2
import yaml
import os
import sys
import time
import shutil
from datetime import datetime
from pathlib import Path
import json

def create_training_structure():
    """
    创建YOLO训练所需的目录结构
    """
    print("=== 创建训练目录结构 ===")
    
    # 创建主要目录
    base_dir = "wave_training_dataset"
    dirs = [
        f"{base_dir}/images/train",
        f"{base_dir}/images/val", 
        f"{base_dir}/images/test",
        f"{base_dir}/labels/train",
        f"{base_dir}/labels/val",
        f"{base_dir}/labels/test",
        f"{base_dir}/raw_videos",
        f"{base_dir}/extracted_frames"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ 创建目录: {dir_path}")
    
    return base_dir

def extract_training_frames(video_path, output_dir, interval_seconds=2):
    """
    从录制的视频中提取训练帧
    
    Args:
        video_path: 视频路径
        interval_seconds: 提取间隔（秒）
    """
    print(f"\n=== 从视频提取训练帧 ===")
    print(f"视频路径: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在")
        return False
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件")
        return False
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"视频属性: {fps:.1f}fps, {total_frames}帧, {duration:.1f}秒")
    
    # 计算帧间隔
    frame_interval = int(fps * interval_seconds)
    expected_frames = total_frames // frame_interval
    
    print(f"提取间隔: {interval_seconds}秒 ({frame_interval}帧)")
    print(f"预计提取: {expected_frames}帧")
    
    extracted_count = 0
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 按间隔提取帧
        if frame_count % frame_interval == 0:
            # 保存帧
            frame_filename = f"frame_{extracted_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            extracted_count += 1
            
            if extracted_count % 10 == 0:
                print(f"已提取 {extracted_count} 帧...")
        
        frame_count += 1
    
    cap.release()
    print(f"✓ 提取完成，共 {extracted_count} 帧")
    return extracted_count

def create_dataset_config(base_dir):
    """
    创建YOLO训练配置文件
    """
    config = {
        'path': os.path.abspath(base_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 2,
        'names': ['wave', 'nowave']
    }
    
    config_path = os.path.join(base_dir, 'data.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✓ 数据集配置文件: {config_path}")
    return config_path

def create_training_script(base_dir):
    """
    创建训练脚本
    """
    script_content = f'''#!/usr/bin/env python3
"""
YOLO挥手检测模型训练脚本
使用自定义数据集重新训练模型
"""

from ultralytics import YOLO
import os

def train_wave_detection_model():
    """训练挥手检测模型"""
    
    # 配置参数
    dataset_config = "{base_dir}/data.yaml"
    
    # 检查数据集配置文件
    if not os.path.exists(dataset_config):
        print(f"错误：数据集配置文件不存在: {{dataset_config}}")
        return False
    
    print("=== 开始训练YOLO挥手检测模型 ===")
    print(f"数据集配置: {{dataset_config}}")
    
    # 加载预训练模型
    model = YOLO('yolov8n.pt')  # 使用轻量级模型，可根据需要改为yolov8s.pt, yolov8m.pt等
    
    # 开始训练
    results = model.train(
        data=dataset_config,           # 数据集配置文件
        epochs=100,                    # 训练轮数，可根据效果调整
        imgsz=640,                     # 输入图像大小
        batch=16,                      # 批次大小，根据显存调整
        patience=20,                   # 早停patience
        save_period=10,                # 每10个epoch保存一次
        device='cuda' if torch.cuda.is_available() else 'cpu',  # 自动选择设备
        workers=4,                     # 数据加载工作进程数
        project='{base_dir}/runs',     # 项目目录
        name='wave_detection',         # 实验名称
        exist_ok=True,                 # 允许覆盖现有实验
        verbose=True,                  # 详细输出
        
        # 数据增强参数
        hsv_h=0.015,                   # 色调增强
        hsv_s=0.7,                     # 饱和度增强
        hsv_v=0.4,                     # 亮度增强
        degrees=10.0,                  # 旋转角度
        translate=0.1,                 # 平移
        scale=0.9,                     # 缩放
        shear=0.0,                     # 剪切
        perspective=0.0,               # 透视变换
        flipud=0.0,                    # 上下翻转
        fliplr=0.5,                    # 左右翻转
        mosaic=1.0,                    # Mosaic增强
        mixup=0.15,                    # Mixup增强
        copy_paste=0.3                 # Copy-paste增强
    )
    
    print("\\n=== 训练完成 ===")
    print(f"最佳模型权重: {{results.save_dir}}/weights/best.pt")
    print(f"最新模型权重: {{results.save_dir}}/weights/last.pt")
    
    # 验证模型
    print("\\n=== 验证模型性能 ===")
    metrics = model.val()
    print(f"mAP50: {{metrics.box.map50:.3f}}")
    print(f"mAP50-95: {{metrics.box.map:.3f}}")
    
    return True

if __name__ == '__main__':
    import torch
    train_wave_detection_model()
'''
    
    script_path = os.path.join(base_dir, 'train_model.py')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"✓ 训练脚本: {script_path}")
    return script_path

def create_annotation_guide():
    """
    创建标注指南
    """
    guide_content = """
# 挥手检测数据标注指南

## 标注类别
- **0**: wave (挥手动作)
- **1**: nowave (非挥手动作/正常状态)

## 标注原则

### Wave (挥手) 类别标注：
1. **手臂抬起**：手臂明显抬起，离开身体
2. **手部可见**：手部在画面中可见
3. **动作意图明确**：明显的挥手意图
4. **包含整个人**：标注框包含进行挥手动作的整个人

### NoWave (非挥手) 类别标注：
1. **正常站立/坐立**：人员处于正常状态
2. **手臂未抬起**：手臂贴近身体或自然垂放
3. **无挥手意图**：无明显挥手动作
4. **包含整个人**：标注框包含整个人

## 标注工具推荐
1. **Labelimg** - 简单易用的图像标注工具
2. **LabelStudio** - 专业的机器学习数据标注平台
3. **CVAT** - 计算机视觉标注工具

## 标注质量要求
1. **边界框准确**：紧贴人物轮廓，不要过大或过小
2. **类别正确**：严格按照上述原则分类
3. **一致性**：同样的动作在不同帧中保持一致的标注
4. **完整性**：确保画面中所有相关人物都被标注

## 数据分布建议
- **训练集**: 70% (约700-1000张图片)
- **验证集**: 20% (约200-300张图片)  
- **测试集**: 10% (约100-150张图片)

每个类别的数据量应该尽可能平衡，避免数据倾斜。

## 标注文件格式
YOLO格式：每行一个目标
```
class_id center_x center_y width height
```
其中坐标都是相对于图片尺寸的归一化值(0-1)。
"""
    
    with open('annotation_guide.md', 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print("✓ 标注指南: annotation_guide.md")

def main():
    """主函数"""
    print("=== YOLO挥手检测模型重新训练准备工具 ===")
    
    # 1. 创建训练目录结构
    base_dir = create_training_structure()
    
    # 2. 从录制的视频中提取帧
    video_paths = [
        r"C:\c\wave_hands\test_videos_20250622_100947\B603_3min_20250622_100957.mp4",
        r"C:\c\wave_hands\test_videos_20250622_100947\B605_3min_20250622_100957.mp4",
        r"C:\c\wave_hands\test_videos_20250622_100947\B606_3min_20250622_100957.mp4"
    ]
    
    total_frames = 0
    for i, video_path in enumerate(video_paths):
        if os.path.exists(video_path):
            camera_name = f"camera_{i+1}"
            output_dir = os.path.join(base_dir, "extracted_frames", camera_name)
            os.makedirs(output_dir, exist_ok=True)
            
            frames = extract_training_frames(video_path, output_dir, interval_seconds=1)
            total_frames += frames
            print(f"✓ {camera_name}: {frames} 帧")
        else:
            print(f"⚠️  视频不存在: {video_path}")
    
    # 3. 创建配置文件
    config_path = create_dataset_config(base_dir)
    
    # 4. 创建训练脚本
    script_path = create_training_script(base_dir)
    
    # 5. 创建标注指南
    create_annotation_guide()
    
    print(f"\n=== 准备完成 ===")
    print(f"✓ 总共提取: {total_frames} 帧")
    print(f"✓ 训练目录: {base_dir}")
    print(f"✓ 配置文件: {config_path}")
    print(f"✓ 训练脚本: {script_path}")
    
    print(f"\n=== 下一步操作 ===")
    print("1. 使用标注工具对提取的帧进行标注")
    print("2. 将标注好的图片和标签文件分别放入对应的train/val/test目录")
    print("3. 运行训练脚本: python train_model.py")
    print("4. 训练完成后，将新模型替换原有的weight/best.pt")
    
    print(f"\n推荐工作流程：")
    print("1. 先标注50-100张图片进行初步训练")
    print("2. 测试初步模型效果")
    print("3. 根据效果继续增加标注数据")
    print("4. 重复训练直到达到理想效果")

if __name__ == '__main__':
    main() 
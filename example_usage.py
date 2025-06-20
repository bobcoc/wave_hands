#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
示例脚本：展示如何使用VideoToYOLODataset类
"""

from video_to_yolo_dataset import VideoToYOLODataset
import os

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 检查是否有示例视频文件
    video_path = "exam_video.mp4"  # 请替换为您的视频文件路径
    
    if not os.path.exists(video_path):
        print(f"请先准备视频文件: {video_path}")
        print("或者修改 video_path 变量为您的视频文件路径")
        return
    
    # 创建转换器
    converter = VideoToYOLODataset(
        video_path=video_path,
        output_dir="basic_dataset",
        model_path="yolov8n.pt"
    )
    
    # 提取帧并检测（默认每200ms一帧）
    num_frames = converter.extract_frames_and_detect()
    
    print(f"基本处理完成，共生成 {num_frames} 张图像")

def example_advanced_usage():
    """高级使用示例"""
    print("\n=== 高级使用示例 ===")
    
    video_path = "exam_video.mp4"  # 请替换为您的视频文件路径
    
    if not os.path.exists(video_path):
        print(f"请先准备视频文件: {video_path}")
        return
    
    # 创建转换器，使用自定义设置
    converter = VideoToYOLODataset(
        video_path=video_path,
        output_dir="advanced_dataset",
        model_path="yolov8s.pt"  # 使用更大的模型获得更好效果
    )
    
    # 高级参数：每500ms一帧，更高的置信度阈值
    num_frames = converter.extract_frames_and_detect(
        interval_ms=500,           # 每500ms提取一帧
        confidence_threshold=0.7   # 更高的置信度阈值
    )
    
    # 可视化前5个检测结果
    converter.visualize_detections(num_samples=5)
    
    print(f"高级处理完成，共生成 {num_frames} 张图像")

def example_custom_processing():
    """自定义处理示例"""
    print("\n=== 自定义处理示例 ===")
    
    video_path = "exam_video.mp4"  # 请替换为您的视频文件路径
    
    if not os.path.exists(video_path):
        print(f"请先准备视频文件: {video_path}")
        return
    
    # 针对监考场景的优化设置
    converter = VideoToYOLODataset(
        video_path=video_path,
        output_dir="exam_monitoring_dataset",
        model_path="yolov8m.pt"  # 中等大小模型，平衡速度和精度
    )
    
    # 监考场景推荐设置
    num_frames = converter.extract_frames_and_detect(
        interval_ms=200,           # 200ms间隔，捕获更多动作
        confidence_threshold=0.4   # 较低阈值，避免遗漏
    )
    
    print(f"监考场景处理完成，共生成 {num_frames} 张图像")
    print("数据集可用于训练监考人员检测模型")

def process_multiple_videos():
    """批量处理多个视频"""
    print("\n=== 批量处理示例 ===")
    
    video_files = [
        "exam_video1.mp4",
        "exam_video2.mp4", 
        "exam_video3.mp4"
    ]
    
    total_frames = 0
    
    for i, video_path in enumerate(video_files):
        if not os.path.exists(video_path):
            print(f"跳过不存在的文件: {video_path}")
            continue
            
        print(f"\n处理第 {i+1} 个视频: {video_path}")
        
        # 为每个视频创建单独的输出目录
        output_dir = f"batch_dataset_video_{i+1}"
        
        converter = VideoToYOLODataset(
            video_path=video_path,
            output_dir=output_dir,
            model_path="yolov8n.pt"
        )
        
        num_frames = converter.extract_frames_and_detect(
            interval_ms=300,
            confidence_threshold=0.5
        )
        
        total_frames += num_frames
        print(f"视频 {i+1} 处理完成，生成 {num_frames} 张图像")
    
    print(f"\n批量处理完成，总共生成 {total_frames} 张图像")

if __name__ == "__main__":
    print("视频到YOLO数据集转换工具 - 使用示例")
    print("=" * 50)
    
    # 运行示例（请根据需要注释/取消注释）
    
    # 基本使用
    # example_basic_usage()
    
    # 高级使用  
    # example_advanced_usage()
    
    # 自定义处理
    # example_custom_processing()
    
    # 批量处理
    # process_multiple_videos()
    
    print("\n注意：请确保有相应的视频文件，并取消注释您想运行的示例函数")
    print("首次运行时，程序会自动下载YOLO模型文件") 
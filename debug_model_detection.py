import cv2
import yaml
import os
import sys
import time
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt

def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def debug_model_detection(video_path, config):
    """
    详细调试模型检测效果
    """
    print("=== 模型检测调试程序 ===")
    
    # 加载配置参数
    weights = config.get('weights', 'weight/best.pt')
    names = config.get('names', ['wave', 'nowave'])
    device = config.get('device', 'cpu')
    
    print(f"模型文件: {weights}")
    print(f"检测类别: {names}")
    print(f"设备: {device}")
    
    # 检查模型文件
    if not os.path.exists(weights):
        print(f"错误：模型文件不存在: {weights}")
        return False
    
    # 加载模型
    try:
        model = YOLO(weights, task='detect')
        model.to(device)
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return False
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频: {video_path}")
        return False
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频属性: {width}x{height}, {fps:.1f}fps, {total_frames}帧")
    
    # 测试不同的置信度阈值
    confidence_levels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
    
    print("\n=== 多置信度阈值测试 ===")
    
    # 随机选择几帧进行测试
    test_frame_indices = np.linspace(0, total_frames-1, min(20, total_frames), dtype=int)
    test_frames = []
    
    # 读取测试帧
    for frame_idx in test_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            test_frames.append((frame_idx, frame.copy()))
    
    print(f"选择了 {len(test_frames)} 帧进行测试")
    
    # 对每个置信度阈值进行测试
    results_summary = {}
    
    for conf_threshold in confidence_levels:
        print(f"\n--- 测试置信度阈值: {conf_threshold} ---")
        
        total_detections = 0
        wave_count = 0
        nowave_count = 0
        frames_with_detection = 0
        
        for frame_idx, frame in test_frames:
            # 进行检测
            results = model(frame, conf=conf_threshold, verbose=False)[0]
            boxes = results.boxes
            
            frame_detections = 0
            if boxes is not None and len(boxes) > 0:
                frames_with_detection += 1
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf_score = float(box.conf[0])
                    
                    if conf_score >= conf_threshold:
                        frame_detections += 1
                        total_detections += 1
                        
                        if cls_id == 0:  # wave
                            wave_count += 1
                        elif cls_id == 1:  # nowave
                            nowave_count += 1
            
            if frame_detections > 0:
                print(f"  帧 {frame_idx}: 检测到 {frame_detections} 个目标")
        
        detection_rate = (frames_with_detection / len(test_frames)) * 100
        
        results_summary[conf_threshold] = {
            'total_detections': total_detections,
            'wave_count': wave_count,
            'nowave_count': nowave_count,
            'frames_with_detection': frames_with_detection,
            'detection_rate': detection_rate
        }
        
        print(f"  总检测数: {total_detections}")
        print(f"  Wave: {wave_count}, NoWave: {nowave_count}")
        print(f"  检测率: {detection_rate:.1f}% ({frames_with_detection}/{len(test_frames)})")
    
    # 输出汇总结果
    print("\n=== 置信度阈值测试汇总 ===")
    print("置信度\t总检测\tWave\tNoWave\t检测率")
    print("-" * 50)
    for conf, result in results_summary.items():
        print(f"{conf:.2f}\t{result['total_detections']}\t{result['wave_count']}\t{result['nowave_count']}\t{result['detection_rate']:.1f}%")
    
    # 如果所有置信度下都没有检测到目标，建议重新训练
    max_detections = max([r['total_detections'] for r in results_summary.values()])
    
    if max_detections == 0:
        print("\n⚠️  所有置信度阈值下都未检测到目标！")
        print("\n这强烈表明需要重新训练模型，可能的原因：")
        print("1. 训练数据与实际场景差异过大")
        print("2. 摄像头角度、光线条件与训练数据不匹配")
        print("3. 人员位置、服装、背景等因素影响")
        print("4. 模型可能损坏或不适用")
        
        print("\n建议的解决方案：")
        print("1. 使用当前摄像头录制的视频重新训练模型")
        print("2. 收集不同光线、角度、人员的训练数据")
        print("3. 进行数据增强以提高模型泛化能力")
        
        return False
    else:
        print(f"\n✓ 最大检测数: {max_detections}")
        best_conf = min([conf for conf, result in results_summary.items() if result['total_detections'] == max_detections])
        print(f"✓ 最佳置信度阈值: {best_conf}")
        
        if max_detections < len(test_frames) * 0.1:  # 检测率低于10%
            print("\n⚠️  检测率过低，仍建议重新训练模型")
            return False
        else:
            print(f"\n✓ 模型在置信度 {best_conf} 下工作正常")
            return True
    
    cap.release()

def visualize_sample_detections(video_path, config, num_samples=5):
    """
    可视化样本检测结果，保存图片用于分析
    """
    print(f"\n=== 保存样本检测图片 ===")
    
    weights = config.get('weights', 'weight/best.pt')
    model = YOLO(weights, task='detect')
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建输出目录
    output_dir = "debug_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    sample_indices = np.linspace(0, total_frames-1, num_samples, dtype=int)
    
    for i, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # 使用极低置信度进行检测
        results = model(frame, conf=0.01, verbose=False)[0]
        
        # 绘制检测结果
        annotated_frame = frame.copy()
        boxes = results.boxes
        
        detection_count = 0
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 绘制所有检测结果，不管置信度
                color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{config.get('names', ['wave', 'nowave'])[cls_id]} {conf_score:.3f}"
                cv2.putText(annotated_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                detection_count += 1
        
        # 保存图片
        sample_path = os.path.join(output_dir, f"sample_{i+1}_frame_{frame_idx}_det_{detection_count}.jpg")
        cv2.imwrite(sample_path, annotated_frame)
        print(f"样本 {i+1}: 帧 {frame_idx}, 检测数: {detection_count}, 保存到: {sample_path}")
    
    cap.release()
    print(f"样本图片保存在: {output_dir}")

def main():
    """主函数"""
    # 使用指定的视频路径
    video_path = r"C:\c\wave_hands\test_videos_20250622_100947\B605_3min_20250622_100957.mp4"
    
    print("=== 模型检测能力调试 ===")
    print(f"视频文件: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在")
        return
    
    # 加载配置
    try:
        config = load_config()
    except Exception as e:
        print(f"读取配置失败: {e}")
        return
    
    # 调试模型检测能力
    model_ok = debug_model_detection(video_path, config)
    
    # 保存样本图片用于人工分析
    visualize_sample_detections(video_path, config, num_samples=10)
    
    # 给出最终建议
    print("\n" + "="*50)
    if not model_ok:
        print("🔴 诊断结果: 需要重新训练模型")
        print("\n建议步骤:")
        print("1. 查看 debug_samples 目录中的样本图片")
        print("2. 确认视频中确实有挥手动作")
        print("3. 如果有动作但检测不到，则需要重新训练")
        print("4. 收集当前场景下的训练数据")
        print("5. 重新训练YOLO模型")
    else:
        print("🟢 诊断结果: 模型工作正常")
        print("建议调整置信度阈值或其他参数")

if __name__ == '__main__':
    main() 
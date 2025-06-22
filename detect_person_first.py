import cv2
import yaml
import os
import sys
import time
import numpy as np
from ultralytics import YOLO
from datetime import datetime

def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def analyze_person_detection(video_path, config, save_samples=True):
    """
    两阶段检测分析：
    1. 使用预训练YOLO检测人物
    2. 在人物区域基础上分析挥手动作
    """
    print("=== 两阶段检测分析 ===")
    print("阶段1: 人物检测 (使用预训练YOLO)")
    print("阶段2: 挥手动作分析 (在人物区域内)")
    
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在: {video_path}")
        return False
    
    # 加载两个模型
    # 1. 预训练的YOLO模型用于人物检测
    try:
        person_model = YOLO('yolov8n.pt')  # 预训练模型，包含person类
        print("✓ 人物检测模型加载成功")
    except Exception as e:
        print(f"✗ 人物检测模型加载失败: {e}")
        return False
    
    # 2. 自定义的挥手检测模型
    weights = config.get('weights', 'weight/best.pt')
    if os.path.exists(weights):
        try:
            wave_model = YOLO(weights, task='detect')
            print("✓ 挥手检测模型加载成功")
            has_wave_model = True
        except Exception as e:
            print(f"⚠️  挥手检测模型加载失败: {e}")
            has_wave_model = False
    else:
        print(f"⚠️  挥手检测模型不存在: {weights}")
        has_wave_model = False
    
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
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\n视频属性:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps:.1f}fps")
    print(f"  总帧数: {total_frames}")
    print(f"  时长: {duration:.1f}秒")
    
    # 统计信息
    stats = {
        'total_frames': 0,
        'frames_with_person': 0,
        'persons_detected': 0,
        'wave_in_person_area': 0,
        'nowave_in_person_area': 0,
        'direct_wave_detection': 0,
        'direct_nowave_detection': 0
    }
    
    # 保存样本的设置
    if save_samples:
        sample_dir = "person_detection_samples"
        os.makedirs(sample_dir, exist_ok=True)
        sample_interval = max(1, total_frames // 20)  # 保存20个样本
        sample_count = 0
    
    print(f"\n开始分析...")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            stats['total_frames'] = frame_count
            
            # 创建展示帧的副本
            display_frame = frame.copy()
            
            # === 阶段1: 人物检测 ===
            person_results = person_model(frame, conf=0.3, verbose=False)[0]
            person_boxes = person_results.boxes
            
            persons_in_frame = []
            
            if person_boxes is not None:
                for box in person_boxes:
                    cls_id = int(box.cls[0])
                    conf_score = float(box.conf[0])
                    
                    # COCO数据集中，person的class_id是0
                    if cls_id == 0 and conf_score >= 0.3:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        persons_in_frame.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf_score
                        })
                        stats['persons_detected'] += 1
                        
                        # 绘制人物检测框（蓝色）
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(display_frame, f"Person {conf_score:.2f}", 
                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            if persons_in_frame:
                stats['frames_with_person'] += 1
            
            # === 阶段2: 在人物区域内进行挥手检测 ===
            wave_detections_in_person = []
            direct_wave_detections = []
            
            if has_wave_model:
                # 对整个画面进行挥手检测
                wave_results = wave_model(frame, conf=0.1, verbose=False)[0]
                wave_boxes = wave_results.boxes
                
                if wave_boxes is not None:
                    for box in wave_boxes:
                        cls_id = int(box.cls[0])
                        conf_score = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        detection_info = {
                            'bbox': [x1, y1, x2, y2],
                            'class_id': cls_id,
                            'confidence': conf_score,
                            'class_name': config.get('names', ['wave', 'nowave'])[cls_id]
                        }
                        
                        # 检查挥手检测是否在人物区域内
                        is_in_person_area = False
                        for person in persons_in_frame:
                            if bbox_overlap(detection_info['bbox'], person['bbox']):
                                is_in_person_area = True
                                wave_detections_in_person.append(detection_info)
                                
                                if cls_id == 0:  # wave
                                    stats['wave_in_person_area'] += 1
                                elif cls_id == 1:  # nowave
                                    stats['nowave_in_person_area'] += 1
                                break
                        
                        if not is_in_person_area:
                            direct_wave_detections.append(detection_info)
                            
                            if cls_id == 0:  # wave
                                stats['direct_wave_detection'] += 1
                            elif cls_id == 1:  # nowave
                                stats['direct_nowave_detection'] += 1
                        
                        # 绘制挥手检测结果
                        if is_in_person_area:
                            color = (0, 255, 0) if cls_id == 0 else (0, 255, 255)  # 绿色(wave) 或 青色(nowave)
                            label = f"✓{detection_info['class_name']} {conf_score:.2f}"
                        else:
                            color = (128, 128, 128)  # 灰色 - 不在人物区域内
                            label = f"✗{detection_info['class_name']} {conf_score:.2f}"
                        
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display_frame, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 在帧上显示统计信息
            info_lines = [
                f"Frame: {frame_count}/{total_frames}",
                f"Persons: {len(persons_in_frame)}",
                f"Wave in Person: {len([d for d in wave_detections_in_person if d['class_id']==0])}",
                f"NoWave in Person: {len([d for d in wave_detections_in_person if d['class_id']==1])}",
                f"Direct Wave: {len([d for d in direct_wave_detections if d['class_id']==0])}",
                f"Direct NoWave: {len([d for d in direct_wave_detections if d['class_id']==1])}"
            ]
            
            for i, line in enumerate(info_lines):
                cv2.putText(display_frame, line, (10, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 保存样本帧
            if save_samples and frame_count % sample_interval == 0:
                sample_path = os.path.join(sample_dir, 
                    f"sample_{sample_count:03d}_frame_{frame_count}_person_{len(persons_in_frame)}_wave_{len(wave_detections_in_person)}.jpg")
                cv2.imwrite(sample_path, display_frame)
                sample_count += 1
                
                print(f"样本 {sample_count}: 帧{frame_count}, 人物:{len(persons_in_frame)}, "
                      f"人物区域内挥手:{len(wave_detections_in_person)}")
            
            # 每100帧输出一次进度
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"进度: {progress:.1f}% - 人物帧:{stats['frames_with_person']}, "
                      f"总人数:{stats['persons_detected']}, "
                      f"人物区域内挥手:{stats['wave_in_person_area'] + stats['nowave_in_person_area']}")
    
    except KeyboardInterrupt:
        print("\n用户中断分析")
    
    cap.release()
    
    # 输出详细统计结果
    print(f"\n=== 分析结果统计 ===")
    print(f"总帧数: {stats['total_frames']}")
    print(f"包含人物的帧数: {stats['frames_with_person']} ({stats['frames_with_person']/stats['total_frames']*100:.1f}%)")
    print(f"检测到的总人数: {stats['persons_detected']}")
    print(f"平均每帧人数: {stats['persons_detected']/stats['total_frames']:.2f}")
    
    print(f"\n=== 挥手检测结果 ===")
    print(f"人物区域内的Wave检测: {stats['wave_in_person_area']}")
    print(f"人物区域内的NoWave检测: {stats['nowave_in_person_area']}")
    print(f"人物区域外的Wave检测: {stats['direct_wave_detection']}")
    print(f"人物区域外的NoWave检测: {stats['direct_nowave_detection']}")
    
    # 分析结果和建议
    person_detection_rate = stats['frames_with_person'] / stats['total_frames']
    wave_in_person_rate = (stats['wave_in_person_area'] + stats['nowave_in_person_area']) / max(stats['persons_detected'], 1)
    
    print(f"\n=== 分析结论 ===")
    
    if person_detection_rate < 0.1:
        print("🔴 人物检测率过低！可能的问题：")
        print("- 摄像头角度不佳")
        print("- 人物在画面中太小")
        print("- 光线条件不好")
        print("- 人物被遮挡")
    elif person_detection_rate < 0.5:
        print("🟡 人物检测率一般，建议优化摄像头位置")
    else:
        print("🟢 人物检测良好")
    
    if stats['wave_in_person_area'] + stats['nowave_in_person_area'] == 0:
        print("🔴 在人物区域内未检测到任何挥手动作！")
        print("强烈建议重新训练挥手检测模型")
    else:
        print("🟢 在人物区域内检测到挥手动作")
        if stats['direct_wave_detection'] + stats['direct_nowave_detection'] > 0:
            print("⚠️  但也有人物区域外的误检测，建议优化模型")
    
    if save_samples:
        print(f"\n样本图片已保存到: {sample_dir}")
    
    return True

def bbox_overlap(bbox1, bbox2, threshold=0.3):
    """
    检查两个边界框是否重叠
    
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
        threshold: 重叠阈值
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # 计算交集
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return False
    
    # 计算交集面积
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # 计算并集面积
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    # 计算IoU
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou >= threshold

def main():
    """主函数"""
    video_path = r"C:\c\wave_hands\test_videos_20250622_100947\B605_3min_20250622_100957.mp4"
    
    print("=== 两阶段检测分析：人物检测 + 挥手识别 ===")
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
    
    # 开始两阶段分析
    success = analyze_person_detection(video_path, config, save_samples=True)
    
    if success:
        print(f"\n=== 改进建议 ===")
        print("1. 如果人物检测率高但挥手检测率低，建议重新训练挥手模型")
        print("2. 可以考虑先裁剪人物区域，再进行挥手检测以提高准确率")
        print("3. 可以结合时序信息，分析连续帧中的动作变化")
        print("4. 可以添加姿态估计来更精确地分析挥手动作")

if __name__ == '__main__':
    main() 
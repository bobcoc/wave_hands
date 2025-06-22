import cv2
import yaml
import os
import sys
import time
from ultralytics import YOLO
from datetime import datetime
import argparse

# 读取配置文件
def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def analyze_video(video_path, config, output_dir=None, show_video=True, save_output=True):
    """
    分析录制的视频，进行挥手检测
    
    Args:
        video_path: 输入视频路径
        config: 配置参数
        output_dir: 输出目录
        show_video: 是否显示视频窗口
        save_output: 是否保存带检测框的输出视频
    """
    
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在: {video_path}")
        return False
    
    print(f"正在分析视频: {video_path}")
    
    # 加载配置参数
    weights = config.get('weights', 'weight/best.pt')
    confidence = float(config.get('confidence', 0.2))
    names = config.get('names', ['wave', 'nowave'])
    device = config.get('device', 'cpu')
    
    print(f"使用模型: {weights}")
    print(f"置信度阈值: {confidence}")
    print(f"检测类别: {names}")
    print(f"设备: {device}")
    
    # 检查模型文件是否存在
    if not os.path.exists(weights):
        print(f"错误：模型文件不存在: {weights}")
        return False
    
    # 加载YOLO模型
    try:
        model = YOLO(weights, task='detect')
        model.to(device)
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件: {video_path}")
        return False
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"视频属性:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps:.2f} fps")
    print(f"  总帧数: {total_frames}")
    print(f"  时长: {duration:.1f}秒")
    
    # 准备输出视频写入器
    output_writer = None
    if save_output and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{video_name}_analyzed_{timestamp}.mp4")
        
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        output_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if output_writer.isOpened():
            print(f"将保存分析结果到: {output_path}")
        else:
            print("警告：无法创建输出视频文件")
            output_writer = None
    
    # 检测统计
    stats = {'wave': 0, 'nowave': 0, 'total_detections': 0, 'frames_with_detection': 0}
    color_map = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255)]
    
    frame_count = 0
    detection_log = []  # 记录每帧的检测结果
    
    print("\n开始分析...")
    print("按 'q' 键退出，按 'p' 键暂停/继续")
    
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                current_time = frame_count / fps if fps > 0 else frame_count
                
                # 进行检测
                detect_start = time.time()
                results = model(frame, conf=confidence, verbose=False)[0]
                detect_time = (time.time() - detect_start) * 1000
                
                # 处理检测结果
                boxes = results.boxes
                frame_detections = []
                frame_wave_count = 0
                frame_nowave_count = 0
                
                if boxes is not None and len(boxes) > 0:
                    stats['frames_with_detection'] += 1
                    
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf_score = float(box.conf[0])
                        
                        if conf_score < confidence:
                            continue
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        color = color_map[cls_id % len(color_map)]
                        
                        # 记录检测结果
                        class_name = names[cls_id] if cls_id < len(names) else f"class_{cls_id}"
                        frame_detections.append({
                            'class': class_name,
                            'confidence': conf_score,
                            'bbox': [x1, y1, x2, y2]
                        })
                        
                        # 统计计数
                        if cls_id == 0:  # wave
                            frame_wave_count += 1
                            stats['wave'] += 1
                        elif cls_id == 1:  # nowave
                            frame_nowave_count += 1
                            stats['nowave'] += 1
                        
                        stats['total_detections'] += 1
                        
                        # 绘制检测框
                        label = f"{class_name} {conf_score:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # 记录当前帧检测日志
                if frame_detections:
                    detection_log.append({
                        'frame': frame_count,
                        'time': current_time,
                        'detections': frame_detections,
                        'wave_count': frame_wave_count,
                        'nowave_count': frame_nowave_count
                    })
                
                # 在视频上显示统计信息
                info_text = f"Frame: {frame_count}/{total_frames} | Time: {current_time:.1f}s | Detect: {detect_time:.1f}ms"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                count_text = f"Wave: {frame_wave_count} | NoWave: {frame_nowave_count}"
                cv2.putText(frame, count_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 保存输出帧
                if output_writer:
                    output_writer.write(frame)
                
                # 每100帧打印一次进度
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"进度: {progress:.1f}% ({frame_count}/{total_frames}) - Wave:{stats['wave']}, NoWave:{stats['nowave']}")
            
            # 显示视频
            if show_video:
                cv2.imshow('Video Analysis', frame)
                key = cv2.waitKey(1 if not paused else 0) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print("暂停" if paused else "继续")
            else:
                # 如果不显示视频，检查是否需要提前退出
                if frame_count % 1000 == 0:
                    print(f"处理进度: {frame_count}/{total_frames}")
    
    except KeyboardInterrupt:
        print("\n用户中断分析")
    
    # 清理资源
    cap.release()
    if output_writer:
        output_writer.release()
    if show_video:
        cv2.destroyAllWindows()
    
    # 输出详细统计结果
    print("\n=== 分析结果 ===")
    print(f"总帧数: {total_frames}")
    print(f"已处理帧数: {frame_count}")
    print(f"有检测结果的帧数: {stats['frames_with_detection']}")
    print(f"检测率: {(stats['frames_with_detection']/frame_count*100):.1f}%")
    print(f"总检测数: {stats['total_detections']}")
    print(f"Wave检测数: {stats['wave']}")
    print(f"NoWave检测数: {stats['nowave']}")
    
    if stats['total_detections'] == 0:
        print("\n⚠️  未检测到任何目标！可能的原因：")
        print("1. 置信度阈值过高 (当前: {:.2f})".format(confidence))
        print("2. 模型不适合当前场景")
        print("3. 视频中确实没有挥手动作")
        print("4. 视频质量问题（光线、角度、清晰度等）")
        print("\n建议：")
        print("- 尝试降低置信度阈值（如0.3或0.1）")
        print("- 检查模型是否正确加载")
        print("- 确认视频中是否有明显的挥手动作")
    else:
        print(f"\n检测成功率: {(stats['frames_with_detection']/frame_count*100):.1f}%")
        if stats['wave'] > 0:
            print(f"Wave/NoWave比例: {stats['wave']/max(stats['nowave'], 1):.2f}")
    
    # 输出详细检测日志（最多显示前10个和后10个）
    if detection_log:
        print(f"\n=== 检测详情 (显示前10个结果) ===")
        for i, log in enumerate(detection_log[:10]):
            print(f"帧 {log['frame']} ({log['time']:.1f}s): Wave={log['wave_count']}, NoWave={log['nowave_count']}")
            for det in log['detections']:
                print(f"  - {det['class']}: {det['confidence']:.3f} at {det['bbox']}")
        
        if len(detection_log) > 10:
            print(f"... (省略 {len(detection_log)-10} 个检测结果)")
    
    print(f"\n分析完成！")
    if output_writer and save_output:
        print(f"带检测框的视频已保存到: {output_path}")
    
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="分析录制的视频，检测挥手动作")
    parser.add_argument("video_path", nargs='?', help="输入视频文件路径")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--output_dir", default="analysis_results", help="输出目录")
    parser.add_argument("--no-display", action="store_true", help="不显示视频窗口")
    parser.add_argument("--no-save", action="store_true", help="不保存输出视频")
    parser.add_argument("--confidence", type=float, help="自定义置信度阈值")
    
    args = parser.parse_args()
    
    # 如果没有提供视频路径参数，使用默认的视频路径
    if not args.video_path:
        # 默认使用用户提到的视频路径
        default_video = r"C:\c\wave_hands\test_videos_20250622_100947\B605_3min_20250622_100957.mp4"
        print(f"使用默认视频路径: {default_video}")
        args.video_path = default_video
    
    print("=== 视频挥手检测分析程序 ===")
    print(f"视频文件: {args.video_path}")
    
    # 加载配置
    try:
        config = load_config(args.config)
        print(f"配置文件: {args.config}")
    except Exception as e:
        print(f"读取配置文件失败: {e}")
        sys.exit(1)
    
    # 如果指定了自定义置信度，覆盖配置文件中的值
    if args.confidence is not None:
        config['confidence'] = args.confidence
        print(f"使用自定义置信度: {args.confidence}")
    
    # 开始分析
    success = analyze_video(
        video_path=args.video_path,
        config=config,
        output_dir=args.output_dir,
        show_video=not args.no_display,
        save_output=not args.no_save
    )
    
    if not success:
        print("分析失败")
        sys.exit(1)
    
    print("\n分析完成！如果检测不到目标，可以尝试：")
    print("1. 降低置信度: python analyze_recorded_video.py --confidence 0.3")
    print("2. 检查模型文件是否正确")
    print("3. 确认视频内容是否包含挥手动作")

if __name__ == '__main__':
    main() 
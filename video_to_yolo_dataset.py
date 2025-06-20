import cv2
import os
import numpy as np
from ultralytics import YOLO
import argparse
from pathlib import Path
import json

class VideoToYOLODataset:
    def __init__(self, video_path, output_dir="yolo_dataset", model_path="yolov8n.pt"):
        """
        初始化视频到YOLO数据集转换器
        
        Args:
            video_path: 输入视频路径
            output_dir: 输出目录
            model_path: YOLO模型路径
        """
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.model_path = model_path
        
        # 创建输出目录结构
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载YOLO模型
        print(f"正在加载YOLO模型: {model_path}")
        self.model = YOLO(model_path)
        
        # 只检测人类 (class 0 in COCO dataset)
        self.person_class_id = 0
        
    def extract_frames_and_detect(self, interval_ms=300, confidence_threshold=0.5):
        """
        从视频中提取帧并进行人物检测
        
        Args:
            interval_ms: 帧提取间隔（毫秒）
            confidence_threshold: 检测置信度阈值
        """
        # 打开视频
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")
        
        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"视频属性: {width}x{height}, {fps} fps, 总帧数: {total_frames}")
        
        # 计算帧间隔
        frame_interval = int((interval_ms / 1000.0) * fps)
        print(f"每 {frame_interval} 帧提取一次 (约 {interval_ms}ms)")
        
        # 计算左下角区域
        crop_width = width // 2+200
        crop_height = height // 2+1800
        crop_x = 300  # 左边
        crop_y = height // 2-800  # 下半部分
        
        print(f"裁剪区域: ({crop_x}, {crop_y}) -> ({crop_x + crop_width}, {crop_y + crop_height})")
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 只在指定间隔处理帧
            if frame_count % frame_interval == 0:
                # 裁剪左下角四分之一
                cropped_frame = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
                
                # 将图像缩小到原来的1/4
                original_height, original_width = cropped_frame.shape[:2]
                new_width = original_width // 4
                new_height = original_height // 4
                cropped_frame = cv2.resize(cropped_frame, (new_width, new_height))
                
                # 输出缩放信息（仅第一次）
                if saved_count == 0:
                    print(f"图像缩放: {original_width}x{original_height} -> {new_width}x{new_height}")
                
                # 保存图像
                image_filename = f"frame1_{saved_count:06d}.jpg"
                image_path = self.images_dir / image_filename
                cv2.imwrite(str(image_path), cropped_frame)
                
                # 使用YOLO检测人物
                results = self.model(cropped_frame, verbose=False)
                
                # 生成YOLO格式的标注
                self._save_yolo_labels(results[0], image_filename, cropped_frame.shape, confidence_threshold)
                
                saved_count += 1
                if saved_count % 10 == 0:
                    print(f"已处理 {saved_count} 帧...")
            
            frame_count += 1
        
        cap.release()
        print(f"完成！共提取 {saved_count} 帧图像")
        
        # 生成数据集配置文件
        self._create_dataset_config()
        
        return saved_count
    
    def _save_yolo_labels(self, result, image_filename, image_shape, confidence_threshold):
        """
        保存YOLO格式的标注文件
        
        Args:
            result: YOLO检测结果
            image_filename: 图像文件名
            image_shape: 图像尺寸 (height, width, channels)
            confidence_threshold: 置信度阈值
        """
        height, width = image_shape[:2]
        label_filename = image_filename.replace('.jpg', '.txt')
        label_path = self.labels_dir / label_filename
        
        labels = []
        
        if result.boxes is not None:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                # 获取类别和置信度
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # 只保存人类检测结果且置信度大于阈值
                if class_id == self.person_class_id and confidence >= confidence_threshold:
                    # 获取边界框坐标 (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # 转换为YOLO格式 (中心点坐标和宽高，归一化)
                    center_x = (x1 + x2) / 2 / width
                    center_y = (y1 + y2) / 2 / height
                    bbox_width = (x2 - x1) / width
                    bbox_height = (y2 - y1) / height
                    
                    # YOLO格式: class_id center_x center_y width height
                    labels.append(f"0 {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}")
        
        # 保存标注文件
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(labels))
    
    def _create_dataset_config(self):
        """创建数据集配置文件"""
        config = {
            "path": str(self.output_dir.absolute()),
            "train": "images",
            "val": "images",  # 这里可以后续分割训练和验证集
            "nc": 1,  # 只有一个类别：人
            "names": ["person"]
        }
        
        config_path = self.output_dir / "dataset.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            import yaml
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"数据集配置文件已保存: {config_path}")
    
    def visualize_detections(self, num_samples=5):
        """
        可视化检测结果
        
        Args:
            num_samples: 要可视化的样本数量
        """
        import matplotlib.pyplot as plt
        
        image_files = list(self.images_dir.glob("*.jpg"))[:num_samples]
        
        fig, axes = plt.subplots(1, len(image_files), figsize=(15, 3))
        if len(image_files) == 1:
            axes = [axes]
        
        for idx, image_file in enumerate(image_files):
            # 读取图像
            image = cv2.imread(str(image_file))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 读取标注
            label_file = self.labels_dir / image_file.name.replace('.jpg', '.txt')
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                height, width = image.shape[:2]
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, center_x, center_y, bbox_width, bbox_height = map(float, parts)
                        
                        # 转换回像素坐标
                        x1 = int((center_x - bbox_width/2) * width)
                        y1 = int((center_y - bbox_height/2) * height)
                        x2 = int((center_x + bbox_width/2) * width)
                        y2 = int((center_y + bbox_height/2) * height)
                        
                        # 绘制边界框
                        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(image_rgb, 'person', (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            axes[idx].imshow(image_rgb)
            axes[idx].set_title(f'Frame {idx}')
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "detection_samples.png", dpi=150, bbox_inches='tight')
        plt.show()
        print(f"检测结果可视化已保存: {self.output_dir / 'detection_samples.png'}")

def main():
    parser = argparse.ArgumentParser(description="从视频生成YOLO人物检测数据集")
    parser.add_argument("video_path", help="输入视频文件路径")
    parser.add_argument("--output_dir", default="yolo_dataset", help="输出目录")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO模型路径")
    parser.add_argument("--interval", type=int, default=200, help="帧提取间隔(毫秒)")
    parser.add_argument("--confidence", type=float, default=0.5, help="检测置信度阈值")
    parser.add_argument("--visualize", action="store_true", help="可视化检测结果")
    
    args = parser.parse_args()
    
    # 检查视频文件是否存在
    if not os.path.exists(args.video_path):
        print(f"错误: 视频文件不存在: {args.video_path}")
        return
    
    try:
        # 创建转换器
        converter = VideoToYOLODataset(
            video_path=args.video_path,
            output_dir=args.output_dir,
            model_path=args.model
        )
        
        # 提取帧并检测
        num_frames = converter.extract_frames_and_detect(
            interval_ms=args.interval,
            confidence_threshold=args.confidence
        )
        
        print(f"\n数据集生成完成!")
        print(f"- 图像数量: {num_frames}")
        print(f"- 图像目录: {converter.images_dir}")
        print(f"- 标注目录: {converter.labels_dir}")
        print(f"- 配置文件: {converter.output_dir / 'dataset.yaml'}")
        
        # 可视化检测结果
        if args.visualize:
            converter.visualize_detections()
        
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main() 
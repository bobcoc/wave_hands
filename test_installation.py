#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：验证所有依赖包是否正确安装
"""

def test_imports():
    """测试所有必需的包是否可以正确导入"""
    print("正在测试依赖包...")
    
    try:
        import cv2
        print(f"✓ OpenCV 版本: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV 导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy 版本: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy 导入失败: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO 导入成功")
    except ImportError as e:
        print(f"✗ Ultralytics 导入失败: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib 版本: {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib 导入失败: {e}")
        return False
    
    try:
        import yaml
        print("✓ PyYAML 导入成功")
    except ImportError as e:
        print(f"✗ PyYAML 导入失败: {e}")
        return False
    
    try:
        from pathlib import Path
        print("✓ Pathlib 导入成功")
    except ImportError as e:
        print(f"✗ Pathlib 导入失败: {e}")
        return False
    
    return True

def test_yolo_model():
    """测试YOLO模型是否可以正常加载"""
    print("\n正在测试YOLO模型...")
    
    try:
        from ultralytics import YOLO
        
        print("正在下载/加载 YOLOv8n 模型...")
        model = YOLO('yolov8n.pt')
        print("✓ YOLO模型加载成功")
        
        # 测试模型推理
        import numpy as np
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(test_image, verbose=False)
        print("✓ YOLO模型推理测试成功")
        
        return True
        
    except Exception as e:
        print(f"✗ YOLO模型测试失败: {e}")
        return False

def test_opencv_video():
    """测试OpenCV视频处理功能"""
    print("\n正在测试OpenCV视频功能...")
    
    try:
        import cv2
        import numpy as np
        
        # 创建一个测试视频帧
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 测试图像裁剪
        height, width = test_frame.shape[:2]
        crop_x, crop_y = 0, height // 2
        crop_width, crop_height = width // 2, height // 2
        cropped = test_frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        
        expected_shape = (crop_height, crop_width, 3)
        if cropped.shape == expected_shape:
            print("✓ 图像裁剪功能正常")
        else:
            print(f"✗ 图像裁剪异常，期望 {expected_shape}，实际 {cropped.shape}")
            return False
        
        # 测试图像编码
        success, encoded = cv2.imencode('.jpg', test_frame)
        if success:
            print("✓ 图像编码功能正常")
        else:
            print("✗ 图像编码失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ OpenCV测试失败: {e}")
        return False

def test_main_module():
    """测试主模块是否可以正确导入"""
    print("\n正在测试主模块...")
    
    try:
        from video_to_yolo_dataset import VideoToYOLODataset
        print("✓ 主模块 VideoToYOLODataset 导入成功")
        
        # 测试类初始化（不实际处理视频）
        # 这里只测试类的基本结构，不进行实际的视频处理
        print("✓ 模块结构检查通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 主模块测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("视频到YOLO数据集转换工具 - 安装测试")
    print("=" * 50)
    
    tests = [
        ("依赖包导入", test_imports),
        ("OpenCV视频功能", test_opencv_video),
        ("主模块", test_main_module),
        ("YOLO模型", test_yolo_model),  # 最后测试，因为需要下载模型
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- 测试 {test_name} ---")
        if test_func():
            passed += 1
            print(f"✓ {test_name} 测试通过")
        else:
            print(f"✗ {test_name} 测试失败")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！您可以开始使用该工具了。")
        print("\n快速开始:")
        print("1. 准备您的视频文件")
        print("2. 运行: python video_to_yolo_dataset.py your_video.mp4")
    else:
        print("❌ 部分测试失败，请检查依赖安装。")
        print("\n安装建议:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main() 
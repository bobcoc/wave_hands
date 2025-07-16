#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试手掌检测程序
"""

import os
import sys
import yaml
import subprocess
import time

def test_config_loading():
    """测试配置文件加载"""
    print("=== 测试配置文件加载 ===")
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("✓ 配置文件加载成功")
        print(f"  模型权重: {config.get('weights', '未设置')}")
        print(f"  检测置信度: {config.get('confidence', '未设置')}")
        print(f"  设备类型: {config.get('device', '未设置')}")
        print(f"  报警目录: {config.get('alarm_dir', '未设置')}")
        
        # 检查视频流配置
        streams = config.get('streams', [])
        print(f"  视频流数量: {len(streams)}")
        if streams:
            print("  视频流列表:")
            for i, stream in enumerate(streams[:3]):  # 只显示前3个
                print(f"    {i+1}. {stream.get('name', 'unnamed')}: {stream.get('url', 'no url')}")
            if len(streams) > 3:
                print(f"    ... 还有 {len(streams)-3} 个视频流")
        
        # 检查本地视频配置
        video_file = config.get('video_file', '')
        if video_file and str(video_file).strip():
            print(f"  本地视频文件: {video_file}")
            if os.path.isfile(video_file):
                print("  ✓ 本地视频文件存在")
            else:
                print("  ✗ 本地视频文件不存在")
        else:
            print("  本地视频文件: 未设置")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        return False

def test_dependencies():
    """测试依赖项"""
    print("\n=== 测试依赖项 ===")
    dependencies = [
        ('cv2', 'opencv-python'),
        ('yaml', 'pyyaml'),
        ('ultralytics', 'ultralytics'),
        ('numpy', 'numpy'),
        ('requests', 'requests')
    ]
    
    all_ok = True
    for module, package in dependencies:
        try:
            __import__(module)
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"✗ {package} 未安装")
            all_ok = False
    
    return all_ok

def test_model_file():
    """测试模型文件"""
    print("\n=== 测试模型文件 ===")
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        weights = config.get('weights', 'weight/best.pt')
        if os.path.isfile(weights):
            print(f"✓ 模型文件存在: {weights}")
            file_size = os.path.getsize(weights) / (1024*1024)  # MB
            print(f"  文件大小: {file_size:.1f} MB")
            return True
        else:
            print(f"✗ 模型文件不存在: {weights}")
            return False
    except Exception as e:
        print(f"✗ 模型文件检查失败: {e}")
        return False

def test_core_module():
    """测试核心检测模块"""
    print("\n=== 测试核心检测模块 ===")
    try:
        from hand_detection_core import load_yolo_model, detect_hands, classify_hand_pose, draw_hand_overlay
        print("✓ 核心检测模块导入成功")
        
        # 测试函数是否存在
        functions = [load_yolo_model, detect_hands, classify_hand_pose, draw_hand_overlay]
        for func in functions:
            if callable(func):
                print(f"✓ 函数 {func.__name__} 可用")
            else:
                print(f"✗ 函数 {func.__name__} 不可用")
                return False
        
        return True
    except ImportError as e:
        print(f"✗ 核心检测模块导入失败: {e}")
        return False

def test_program_structure():
    """测试程序结构"""
    print("\n=== 测试程序结构 ===")
    programs = [
        ('detect_video.py', '视频流检测程序'),
        ('detect_local_video.py', '本地视频检测程序'),
        ('hand_detection_core.py', '核心检测模块')
    ]
    
    all_ok = True
    for program, description in programs:
        if os.path.isfile(program):
            print(f"✓ {description} 存在: {program}")
        else:
            print(f"✗ {description} 不存在: {program}")
            all_ok = False
    
    return all_ok

def test_alarm_directory():
    """测试报警目录"""
    print("\n=== 测试报警目录 ===")
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        alarm_dir = config.get('alarm_dir', 'alarms')
        if os.path.exists(alarm_dir):
            print(f"✓ 报警目录存在: {alarm_dir}")
        else:
            print(f"⚠ 报警目录不存在，将自动创建: {alarm_dir}")
            try:
                os.makedirs(alarm_dir, exist_ok=True)
                print(f"✓ 报警目录创建成功: {alarm_dir}")
            except Exception as e:
                print(f"✗ 报警目录创建失败: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ 报警目录检查失败: {e}")
        return False

def main():
    """主测试函数"""
    print("手掌检测程序测试")
    print("=" * 50)
    
    tests = [
        test_config_loading,
        test_dependencies,
        test_model_file,
        test_core_module,
        test_program_structure,
        test_alarm_directory
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ 测试异常: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    
    test_names = [
        "配置文件加载",
        "依赖项检查",
        "模型文件检查",
        "核心模块检查",
        "程序结构检查",
        "报警目录检查"
    ]
    
    all_passed = True
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {i+1}. {name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 所有测试通过！程序可以正常使用。")
        print("\n使用方法:")
        print("  视频流检测: python detect_video.py")
        print("  本地视频检测: python detect_local_video.py")
    else:
        print("❌ 部分测试失败，请检查上述问题。")
        print("\n常见解决方案:")
        print("  1. 安装缺失的依赖: pip install opencv-python ultralytics pyyaml numpy requests")
        print("  2. 确保模型文件 weight/best.pt 存在")
        print("  3. 检查配置文件 config.yaml 格式是否正确")
    
    return all_passed

if __name__ == '__main__':
    main() 
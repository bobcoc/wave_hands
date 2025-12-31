#!/usr/bin/env python3
"""
GStreamer 支持检测工具

用于检测当前 OpenCV 是否支持 GStreamer 后端，以及测试 RTSP 流的 GStreamer 管道。
GStreamer 后端可以真正实现 max-buffers=1 + drop=true，从管道层自动丢弃旧帧，
避免缓冲区积累，是处理 RTSP 流最有效的缓冲区控制方案。

用法:
    python check_gstreamer.py                    # 检测 GStreamer 支持
    python check_gstreamer.py rtsp://xxx/stream  # 测试 RTSP 流
"""

import cv2
import sys
import time


def check_gstreamer_support():
    """
    检测当前 OpenCV 是否支持 GStreamer 后端
    返回: (是否支持, 详细信息字典)
    """
    # 安全获取 OpenCV 版本
    try:
        opencv_ver = cv2.__version__
    except AttributeError:
        try:
            opencv_ver = cv2.getVersionString()
        except Exception:
            opencv_ver = "未知"
    
    result = {
        'opencv_version': opencv_ver,
        'build_support': False,
        'backend_available': False,
        'pipeline_test': False,
        'error': None
    }
    
    try:
        # 方法1: 检查 OpenCV 编译信息
        build_info = cv2.getBuildInformation()
        result['build_support'] = 'GStreamer:                   YES' in build_info
        
        # 方法2: 尝试获取后端列表（OpenCV 4.x）
        if hasattr(cv2, 'videoio_registry'):
            try:
                backends = cv2.videoio_registry.getBackends()
                result['backend_available'] = cv2.CAP_GSTREAMER in backends
                
                # 获取后端名称列表
                backend_names = []
                for b in backends:
                    try:
                        name = cv2.videoio_registry.getBackendName(b)
                        backend_names.append(name)
                    except Exception:
                        backend_names.append(str(b))
                result['available_backends'] = backend_names
            except Exception as e:
                result['backend_available'] = result['build_support']
                result['registry_error'] = str(e)
        else:
            result['backend_available'] = result['build_support']
        
        # 方法3: 实际测试 GStreamer 管道（最可靠）
        test_pipeline = "videotestsrc num-buffers=1 ! videoconvert ! appsink"
        test_cap = cv2.VideoCapture(test_pipeline, cv2.CAP_GSTREAMER)
        result['pipeline_test'] = test_cap.isOpened()
        if test_cap:
            test_cap.release()
            
    except Exception as e:
        result['error'] = str(e)
    
    return result['pipeline_test'], result


def create_gstreamer_rtsp_pipeline(url, max_buffers=1, drop=True, latency=0):
    """
    创建用于 RTSP 流的 GStreamer 管道字符串
    
    参数:
        url: RTSP 流地址
        max_buffers: appsink 最大缓冲帧数（默认1，只保留最新帧）
        drop: 是否丢弃旧帧（默认True）
        latency: 延迟设置（毫秒，默认0表示最低延迟）
    
    返回:
        GStreamer 管道字符串
    """
    drop_str = "true" if drop else "false"
    pipeline = (
        f'rtspsrc location="{url}" latency={latency} ! '
        f'decodebin ! '
        f'videoconvert ! '
        f'appsink max-buffers={max_buffers} drop={drop_str} sync=false'
    )
    return pipeline


def test_rtsp_stream(url, duration=10):
    """
    测试 RTSP 流的 GStreamer 管道
    
    参数:
        url: RTSP 流地址
        duration: 测试时长（秒）
    """
    print(f"\n{'='*60}")
    print(f"测试 RTSP 流: {url}")
    print(f"{'='*60}")
    
    pipeline = create_gstreamer_rtsp_pipeline(url)
    print(f"\nGStreamer 管道:\n{pipeline}\n")
    
    print("正在连接...")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("❌ GStreamer 管道打开失败!")
        print("\n可能的原因:")
        print("  1. OpenCV 未编译 GStreamer 支持")
        print("  2. 系统未安装 GStreamer 库")
        print("  3. RTSP 地址不正确或网络不通")
        print("\n尝试使用 FFmpeg 后端...")
        
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            print("✓ FFmpeg 后端可用（但缓冲区控制无效）")
        else:
            print("❌ FFmpeg 后端也失败")
        if cap:
            cap.release()
        return False
    
    print("✓ GStreamer 管道打开成功!")
    
    # 获取流信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"流信息: {width}x{height} @ {fps}fps")
    
    # 测试读取帧
    print(f"\n开始测试读取帧（持续 {duration} 秒）...")
    print("观察 read 耗时是否稳定，若持续很短说明缓冲区控制有效\n")
    
    start_time = time.time()
    frame_count = 0
    read_times = []
    
    while time.time() - start_time < duration:
        read_start = time.time()
        ret, frame = cap.read()
        read_time = time.time() - read_start
        
        if not ret:
            print("读取帧失败，流可能断开")
            break
        
        frame_count += 1
        read_times.append(read_time)
        
        # 每秒打印一次统计
        if frame_count % 25 == 0:
            avg_read = sum(read_times[-25:]) / len(read_times[-25:])
            print(f"帧 {frame_count:4d}: read耗时 {read_time*1000:6.1f}ms, 近25帧平均: {avg_read*1000:.1f}ms")
    
    cap.release()
    
    # 统计结果
    elapsed = time.time() - start_time
    actual_fps = frame_count / elapsed if elapsed > 0 else 0
    avg_read_time = sum(read_times) / len(read_times) if read_times else 0
    max_read_time = max(read_times) if read_times else 0
    
    print(f"\n{'='*60}")
    print("测试结果:")
    print(f"  总帧数: {frame_count}")
    print(f"  实际FPS: {actual_fps:.1f}")
    print(f"  平均read耗时: {avg_read_time*1000:.1f}ms")
    print(f"  最大read耗时: {max_read_time*1000:.1f}ms")
    
    if avg_read_time < 0.1:  # 平均小于100ms
        print("\n✓ GStreamer 缓冲区控制有效！read 耗时很短，说明总是获取最新帧")
    else:
        print("\n⚠ read 耗时较长，可能有缓冲积压")
    
    return True


def print_gstreamer_install_guide():
    """打印 GStreamer 安装指南"""
    print("\n" + "="*60)
    print("GStreamer 安装指南")
    print("="*60)
    
    print("""
【macOS】
    brew install gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly
    
    然后需要重新编译 OpenCV with GStreamer:
    pip uninstall opencv-python opencv-python-headless
    pip install opencv-python --no-binary :all:
    
    或使用 conda:
    conda install -c conda-forge opencv gstreamer

【Ubuntu/Debian】
    sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
    sudo apt-get install gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
    
    然后重新编译 OpenCV 或使用预编译版本:
    pip install opencv-python  # 部分版本已包含 GStreamer

【Windows】
    1. 下载并安装 GStreamer: https://gstreamer.freedesktop.org/download/
    2. 安装时选择 "Complete" 安装类型
    3. 将 GStreamer bin 目录添加到 PATH
    4. 重新安装 OpenCV 或使用包含 GStreamer 的预编译版本

【验证安装】
    安装后重新运行此脚本验证 GStreamer 是否可用
""")


def main():
    print("="*60)
    print("GStreamer 支持检测工具")
    print("="*60)
    
    # 检测 GStreamer 支持
    supported, info = check_gstreamer_support()
    
    print(f"\nOpenCV 版本: {info['opencv_version']}")
    print(f"编译时 GStreamer 支持: {'✓ 是' if info['build_support'] else '✗ 否'}")
    print(f"GStreamer 后端可用: {'✓ 是' if info['backend_available'] else '✗ 否'}")
    print(f"GStreamer 管道测试: {'✓ 通过' if info['pipeline_test'] else '✗ 失败'}")
    
    if 'available_backends' in info:
        print(f"\n可用视频后端: {', '.join(info['available_backends'])}")
    
    if info.get('error'):
        print(f"\n错误信息: {info['error']}")
    
    if supported:
        print("\n" + "="*60)
        print("✓ GStreamer 可用！")
        print("="*60)
        print("""
GStreamer 后端的优势:
  • appsink 可以真正实现 max-buffers=1 + drop=true
  • 从管道层自动丢弃旧帧，不会积累缓冲
  • 不会触发 FFmpeg 那种长时间阻塞
  • 是 RTSP 流缓冲区控制的最佳方案

已在 config.yaml 中默认启用 GStreamer 后端。
""")
    else:
        print("\n" + "="*60)
        print("✗ GStreamer 不可用")
        print("="*60)
        print("\n系统将回退到 FFmpeg 后端（缓冲区设置无效，需手动 grab 清空）")
        print_gstreamer_install_guide()
    
    # 如果提供了 RTSP URL，测试流
    if len(sys.argv) > 1:
        rtsp_url = sys.argv[1]
        if rtsp_url.startswith('rtsp://'):
            test_rtsp_stream(rtsp_url)
        else:
            print(f"\n警告: '{rtsp_url}' 不是有效的 RTSP URL")
    else:
        print("\n提示: 可以通过命令行参数测试 RTSP 流:")
        print("  python check_gstreamer.py rtsp://username:password@ip:port/path")


if __name__ == '__main__':
    main()

# 检查PyTorch与GPU环境的工具脚本
# 用途：检测当前Python环境是否已安装PyTorch，并判断是否支持NVIDIA GPU加速，输出详细诊断信息。

import sys

try:
    import torch
except ImportError:
    print("未检测到PyTorch库，请先安装PyTorch。\n安装方法示例：pip install torch 或参考 https://pytorch.org/get-started/locally/")
    sys.exit(1)

print(f"PyTorch 版本: {torch.__version__}")
print(f"PyTorch 编译时CUDA版本: {getattr(torch.version, 'cuda', '未知')}")

cuda_available = torch.cuda.is_available()
if not cuda_available:
    print("未检测到可用的NVIDIA GPU，或当前PyTorch未启用CUDA支持。\n\n常见原因：\n1. 您的电脑未安装NVIDIA显卡驱动或驱动版本过低。\n2. 当前PyTorch为CPU版本，未包含CUDA支持。\n3. CUDA Toolkit未正确安装或环境变量未配置。\n4. 物理GPU资源被占用或损坏。\n\n建议：\n- 请访问 https://pytorch.org/get-started/locally/ 选择合适的CUDA版本重新安装PyTorch。\n- 检查NVIDIA驱动和CUDA Toolkit安装情况。\n- 如有多块显卡，确保至少有一块可用。")
    sys.exit(2)

num_gpus = torch.cuda.device_count()
print(f"检测到 {num_gpus} 块可用的NVIDIA GPU：")
for i in range(num_gpus):
    name = torch.cuda.get_device_name(i)
    print(f"  GPU {i}: {name}")

# 可选：尝试获取CUDA驱动版本
try:
    cuda_driver_version = torch._C._cuda_getDriverVersion()
    print(f"CUDA 驱动版本: {cuda_driver_version // 1000}.{(cuda_driver_version % 1000) // 10}")
except Exception:
    print("无法获取CUDA驱动版本信息。")

print("\n环境检测完成，您的PyTorch已支持GPU加速，可放心运行detect_video.py。") 
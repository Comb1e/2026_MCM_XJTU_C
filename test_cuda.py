import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"GPU 数量: {torch.cuda.device_count()}")
print(f"当前 GPU: {torch.cuda.current_device()}")
print(f"GPU 名称: {torch.cuda.get_device_name(0)}")

# 测试 GPU 计算
if torch.cuda.is_available():
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    y = torch.tensor([4.0, 5.0, 6.0]).cuda()
    z = x + y
    print(f"GPU 计算测试: {z}")
    print(f"张量在: {z.device}")
else:
    print("CUDA 不可用，请检查安装")
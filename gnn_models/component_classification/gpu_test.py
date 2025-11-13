import torch
import torch_geometric

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Test GPU speed
if torch.cuda.is_available():
    device = torch.device('cuda')
    x = torch.randn(10000, 10000).to(device)
    y = torch.randn(10000, 10000).to(device)
    
    import time
    start = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    end = time.time()
    
    print(f"GPU matrix multiplication time: {end - start:.4f} seconds")
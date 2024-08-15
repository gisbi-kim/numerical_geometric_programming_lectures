import torch
import time

# CUDA 사용 가능 여부 확인
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

# 비교를 위한 텐서 크기
tensor_size = (10000, 10000)

# CPU에서 연산
start_time = time.time()
cpu_tensor = torch.randn(tensor_size)
cpu_result = cpu_tensor @ cpu_tensor.T  # 행렬 곱
cpu_time = time.time() - start_time
print(f"CPU computation time: {cpu_time:.4f} seconds")

# CUDA(GPU)에서 연산
if torch.cuda.is_available():
    start_time = time.time()
    gpu_tensor = torch.randn(tensor_size, device=device)
    gpu_result = gpu_tensor @ gpu_tensor.T  # 행렬 곱
    gpu_time = time.time() - start_time
    print(f"GPU computation time: {gpu_time:.4f} seconds")
else:
    gpu_time = None
    print("GPU computation not performed.")

# 결과 비교 출력
if gpu_time is not None:
    print(f"Time difference (CPU - GPU): {cpu_time - gpu_time:.4f} seconds")
else:
    print("GPU time is not available, skipping time difference computation.")

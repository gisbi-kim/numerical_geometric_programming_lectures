import tensorflow as tf
import time

# TensorFlow GPU 장치 확인
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs available:", gpus)
else:
    print("No GPUs available.")

# 비교를 위한 텐서 크기
matrix_size = 10000

# CPU에서 연산
with tf.device('/CPU:0'):
    start_time = time.time()
    cpu_matrix1 = tf.random.normal([matrix_size, matrix_size])
    cpu_matrix2 = tf.random.normal([matrix_size, matrix_size])
    cpu_result = tf.matmul(cpu_matrix1, cpu_matrix2)
    cpu_time = time.time() - start_time
    print(f"CPU computation time: {cpu_time:.4f} seconds")

# GPU에서 연산
if gpus:
    with tf.device('/GPU:0'):
        start_time = time.time()
        gpu_matrix1 = tf.random.normal([matrix_size, matrix_size])
        gpu_matrix2 = tf.random.normal([matrix_size, matrix_size])
        gpu_result = tf.matmul(gpu_matrix1, gpu_matrix2)
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

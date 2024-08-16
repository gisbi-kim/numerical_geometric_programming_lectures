#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <tbb/tbb.h>

// CUDA 커널
__global__ void sum_with_cuda_kernel(unsigned long long *sum, const long size) {
  unsigned long long temp_sum = 0;
  const int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  for (long i = tid; i < size; i += blockDim.x * gridDim.x) {
    temp_sum += i;
  }
  atomicAdd(sum, temp_sum);
}

// CUDA 합계 함수
unsigned long long sum_with_cuda(const long size, const int blocks,
                                 const int threads_per_block) {
  unsigned long long *dev_sum;
  unsigned long long host_sum = 0;

  cudaMalloc((void **)&dev_sum, sizeof(unsigned long long));
  cudaMemcpy(dev_sum, &host_sum, sizeof(unsigned long long),
             cudaMemcpyHostToDevice);

  sum_with_cuda_kernel<<<blocks, threads_per_block>>>(dev_sum, size);
  cudaDeviceSynchronize();

  cudaMemcpy(&host_sum, dev_sum, sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);
  cudaFree(dev_sum);

  return host_sum;
}

void print_gpu_memory_usage() {
  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  size_t used_mem = total_mem - free_mem;

  std::cout << "GPU VRAM Usage: " << used_mem / (1024.0 * 1024.0)
            << " MB out of " << total_mem / (1024.0 * 1024.0) << " MB"
            << std::endl;
}

int main(int argc, char *argv[]) {
  // init values
  long size = 0;
  int num_threads = 0;
  int cuda_blocks = 0;
  int cuda_threads_per_block = 0;
  bool skip_single_thread = false;

  // parse args
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg.find("--size=") == 0) {
      size = std::stol(arg.substr(7));
    } else if (arg.find("--num_thread=") == 0) {
      num_threads = std::stoi(arg.substr(13));
    } else if (arg.find("--cuda_blocks=") == 0) {
      cuda_blocks = std::stoi(arg.substr(14));
    } else if (arg.find("--cuda_threads_per_block=") == 0) {
      cuda_threads_per_block = std::stoi(arg.substr(26));
    } else if (arg == "--skip_single_thread=true") {
      skip_single_thread = true;
    } else if (arg == "--skip_single_thread=false") {
      skip_single_thread = false;
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return 1;
    }
  }

  // check correctness of the args
  if (size <= 0 || num_threads <= 0 || cuda_blocks <= 0 ||
      cuda_threads_per_block <= 0) {
    std::cerr << "Usage: " << argv[0]
              << " --size=<size> --skip_single_thread=true "
                 "--num_thread=<num_threads> --cuda_blocks=<blocks> "
                 "--cuda_threads_per_block=<threads_per_block>"
              << std::endl;
    return 1;
  }

  // Time cost of non-TBB
  if (!skip_single_thread) {
    unsigned long long sum_no_tbb = 0;

    auto start_no_tbb = std::chrono::high_resolution_clock::now();

    for (long i = 0; i < size; i++) {
      sum_no_tbb += i;
    }

    auto end_no_tbb = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_no_tbb = end_no_tbb - start_no_tbb;

    std::cout << " " << std::endl;
    std::cout << "Sum without TBB or CUDA: " << sum_no_tbb << std::endl;
    std::cout << "Time without TBB or CUDA: " << duration_no_tbb.count()
              << " seconds" << std::endl;
  }

  // Set the number of thread to be used
  {
    std::atomic<unsigned long long> sum_with_tbb(0);

    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
                                num_threads);
    const int max_threads = tbb::this_task_arena::max_concurrency();
    std::cout << " " << std::endl;
    std::cout << "Number of threads TBB is using: " << num_threads << "/"
              << max_threads << std::endl;

    // Time cost of TBB
    const auto start_with_tbb = std::chrono::high_resolution_clock::now();

    tbb::parallel_for(tbb::blocked_range<long>(0, size),
                      [&](const tbb::blocked_range<long> &r) {
                        unsigned long long temp_sum = 0;
                        for (long i = r.begin(); i != r.end(); ++i) {
                          temp_sum += i;
                        }
                        sum_with_tbb.fetch_add(temp_sum,
                                               std::memory_order_relaxed);
                      });

    const auto end_with_tbb = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_with_tbb =
        end_with_tbb - start_with_tbb;

    std::cout << "Sum with TBB: " << sum_with_tbb << std::endl;
    std::cout << "Time with TBB: " << duration_with_tbb.count() << " seconds"
              << std::endl;
  }

  // Time cost of CUDA
  {
    const auto start_with_cuda = std::chrono::high_resolution_clock::now();

    const unsigned long long sum_cuda =
        sum_with_cuda(size, cuda_blocks, cuda_threads_per_block);

    const auto end_with_cuda = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> duration_with_cuda =
        end_with_cuda - start_with_cuda;

    std::cout << " " << std::endl;
    std::cout << "Sum with CUDA: " << sum_cuda << std::endl;
    std::cout << "Time with CUDA: " << duration_with_cuda.count() << " seconds"
              << std::endl;

    // Print GPU memory usage
    print_gpu_memory_usage();
  }

  return 0;
}

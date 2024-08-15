#include <iostream>
#include <chrono>
#include <tbb/tbb.h>
#include <atomic>
#include <string>

int main(int argc, char* argv[]) {
    // init values
    long size = 0;
    int num_threads = 0;

    // parse args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg.find("--size=") == 0) {
            size = std::stol(arg.substr(7));
        } else if (arg.find("--num_thread=") == 0) {
            num_threads = std::stoi(arg.substr(13));
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 1;
        }
    }

    // check correctness of the args
    if (size <= 0 || num_threads <= 0) {
        std::cerr << "Usage: " << argv[0] << " --size=<size> --num_thread=<num_threads>" << std::endl;
        return 1;
    }

    unsigned long long sum_no_tbb = 0;
    std::atomic<unsigned long long> sum_with_tbb(0);

    // time cost of non-tbb
    auto start_no_tbb = std::chrono::high_resolution_clock::now();

    for (long i = 0; i < size; i++) {
        sum_no_tbb += i;
    }

    auto end_no_tbb = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_no_tbb = end_no_tbb - start_no_tbb;

    std::cout << " " << std::endl;
    std::cout << "Sum without TBB: " << sum_no_tbb << std::endl;
    std::cout << "Time without TBB: " << duration_no_tbb.count() << " seconds" << std::endl;

    // set the number of thread to be used 
    tbb::global_control control(tbb::global_control::max_allowed_parallelism, num_threads);
    int max_threads = tbb::this_task_arena::max_concurrency();
    std::cout << " " << std::endl;
    std::cout << "Number of threads TBB is using: " << num_threads << "/" << max_threads << std::endl;

    // time cost of tbb
    auto start_with_tbb = std::chrono::high_resolution_clock::now();

    tbb::parallel_for(tbb::blocked_range<long>(0, size),
        [&](const tbb::blocked_range<long>& r) {
            unsigned long long temp_sum = 0;
            for (long i = r.begin(); i != r.end(); ++i) {
                temp_sum += i;
            }
            sum_with_tbb.fetch_add(temp_sum, std::memory_order_relaxed);
        });

    auto end_with_tbb = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_with_tbb = end_with_tbb - start_with_tbb;

    std::cout << "Sum with TBB: " << sum_with_tbb << std::endl;
    std::cout << "Time with TBB: " << duration_with_tbb.count() << " seconds" << std::endl;

    return 0;
}

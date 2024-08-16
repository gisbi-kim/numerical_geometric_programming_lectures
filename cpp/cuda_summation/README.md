## How to do
- HW/SW preparation 
    - You need to be equipped with a GPU machine. 
    - Then, do `docker-build.sh` script. For me, I use the base image `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel`, which already has cuda APIs.
    - Then, start a docker container using `docker-run.sh` (change the src path for yours). 
- build script 

        $ nvcc -o main main.cu -ltbb 

    or 

        $ mkdir build; cd build; cmake ..; make -j; 

- run script 
        
        $ ./main --size=200000000000 --skip_single_thread=false --num_thread=16 --cuda_blocks=256 --cuda_threads_per_block=256

- The result (RTX 4060 laptop)

        root@82887fbc2c43:/test/cpp/cuda_summation# ./main --size=200000000000 --skip_single_thread=false --num_thread=16 --cuda_blocks=256 --cuda_threads_per_block=256
        
        Sum without TBB or CUDA: 3729423998846048256
        Time without TBB or CUDA: 303.894 seconds

        Number of threads TBB is using: 16/16
        Sum with TBB: 3729423998846048256
        Time with TBB: 36.0692 seconds
        
        Sum with CUDA: 3729423998846048256
        Time with CUDA: 2.056 seconds

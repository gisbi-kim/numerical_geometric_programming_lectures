# gpu_checker
- simple programs of pytorch and tf

## example env 
- rtx 4060 laptop

      PS C:\Users\gskim> nvidia-smi.exe
      Sun Aug 11 18:21:48 2024
      +-----------------------------------------------------------------------------------------+
      | NVIDIA-SMI 555.97                 Driver Version: 555.97         CUDA Version: 12.5     |
      |-----------------------------------------+------------------------+----------------------+
      | GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
      | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
      |                                         |                        |               MIG M. |
      |=========================================+========================+======================|
      |   0  NVIDIA GeForce RTX 4060 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
      | N/A   52C    P3             13W /   42W |       0MiB /   8188MiB |      0%      Default |
      |                                         |                        |                  N/A |
      +-----------------------------------------+------------------------+----------------------+
      
      +-----------------------------------------------------------------------------------------+
      | Processes:                                                                              |
      |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
      |        ID   ID                                                               Usage      |
      |=========================================================================================|
      |  No running processes found                                                             |
      +-----------------------------------------------------------------------------------------+


## docker run commands 
- pytorch 
  > docker run --gpus all --network host --rm -it --name my_pytorch_container pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

- tensorflow
  > docker run --gpus all --network host --rm -it --name tf_gpu_container tensorflow/tensorflow:2.13.0-gpu bash

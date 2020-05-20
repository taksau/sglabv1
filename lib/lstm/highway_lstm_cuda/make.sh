#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

# Which CUDA capabilities do we want to pre-build for?
# https://developer.nvidia.com/cuda-gpus
#   Compute/shader model   Cards
#   61                    P4, P40, Titan X
#   60                    P100
#   52                    M40
#   37                    K80
#   35                    K40, K20
#   30                    K10, Grid K520 (AWS G2)


# Nvidia doesn't guarantee binary compatability across GPU versions.
# However, binary compatibility within one GPU generation can be guaranteed
# under certain conditions because they share the basic instruction set.
# This is the case between two GPU versions that do not show functional 
# differences at all (for instance when one version is a scaled down version
# of the other), or when one version is functionally included in the other.

# To fix this problem, we can create a 'fat binary' which generates multiple
# translations of the CUDA source. The most appropriate version is chosen at
# runtime by the CUDA driver. See:
# http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-compilation
# http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#fatbinaries



echo "Building kernel for following target architectures: "

cd src
echo "Compiling kernel"
/usr/local/cuda/bin/nvcc -c -o highway_lstm_kernel.cu.o highway_lstm_kernel.cu --compiler-options -fPIC -gencode arch=compute_70,code=sm_70
cd ../
python build.py

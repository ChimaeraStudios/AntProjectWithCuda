cmake_minimum_required(VERSION 3.29)
project(AntProject CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(AntProject src/main.cu
        src/kernel.cu
        src/randomStatesKernel.cu
)

set_target_properties(AntProject PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_ARCHITECTURES 86)
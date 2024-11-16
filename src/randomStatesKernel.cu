//
// Created by Andrea Menci on 15/11/2024.
//

#include "randomStatesKernel.cuh"
#include <cstdio>

__global__ void initializeRandomStates(curandState* states, int seed, int numAnts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numAnts) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}


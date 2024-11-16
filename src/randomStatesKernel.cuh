//
// Created by Andrea Menci on 15/11/2024.
//

#ifndef RANDOMSTATESKERNEL_CUH
#define RANDOMSTATESKERNEL_CUH
#include <curand_kernel.h>

__global__ void initializeRandomStates(curandState* states, int seed, int numAnts);

#endif //RANDOMSTATESKERNEL_CUH

#ifndef KERNEL_CUH
#define KERNEL_CUH
#include <curand_kernel.h>
#include <cstdio>


struct Ant {
    int x, y;
    bool hasFood;
};

void initializeEnvironment(int* environment, int width, int height, int numFoodSources);
void initializeAnts(Ant* ants, int numAnts, int width, int height);

__global__ void moveAnts(Ant* ants, int* environment, curandState* states, int numAnts, int width, int height);

#endif

//
// Created by Andrea Menci on 15/11/2024.
//

/*
Ogni cella può contenere:
0: Vuoto
1: Cibo
3: Nido.
*/

#include "kernel.cuh"

void initializeEnvironment(int* environment, int width, int height, int numFoodSources) {
    for (int i = 0; i < width * height; i++) {
        environment[i] = 0;
    }

    environment[width / 2 + height / 2 * width] = 3;

    srand(time(nullptr));
    for (int i = 0; i < numFoodSources; i++) {
        int x, y;

        do {
            x = rand() % width;
            y = rand() % height;
        } while (environment[y * width + x] != 0);

        environment[y * width + x] = 1;
    }
}

void initializeAnts(Ant* ants, int numAnts, int width, int height) {
    for (int i = 0; i < numAnts; i++) {
        ants[i].x = width / 2;
        ants[i].y = height / 2;
        ants[i].hasFood = false;
    }
}

__device__ bool isWithinRange(int x1, int y1, int x2, int y2, int range) {
    int dx = x2 - x1;
    int dy = y2 - y1;
    return dx * dx + dy * dy <= range * range;
}

__global__ void moveAnts(Ant* ants, int* environment, curandState* states, int numAnts, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numAnts) {
        Ant* ant = &ants[idx];

        int hiveX = width / 2;
        int hiveY = height / 2;

        int senseRange = 0.2 * max(width, height);
        int dx = 0, dy = 0;

        if (ant->hasFood) {
            dx = (hiveX > ant->x) - (hiveX < ant->x);
            dy = (hiveY > ant->y) - (hiveY < ant->y);

            dx += static_cast<int>(curand_uniform(&states[idx]) * 3) - 1;
            dy += static_cast<int>(curand_uniform(&states[idx]) * 3) - 1;
        } else {
            bool foodFound = false;

            for (int offsetX = -senseRange; offsetX <= senseRange && !foodFound; offsetX++) {
                for (int offsetY = -senseRange; offsetY <= senseRange && !foodFound; offsetY++) {
                    int newX = ant->x + offsetX;
                    int newY = ant->y + offsetY;

                    if (newX >= 0 && newX < width && newY >= 0 && newY < height &&
                        isWithinRange(ant->x, ant->y, newX, newY, senseRange) &&
                        environment[newY * width + newX] == 1) {
                        dx = (newX > ant->x) - (newX < ant->x);
                        dy = (newY > ant->y) - (newY < ant->y);
                        foodFound = true;
                    }
                }
            }

            if (foodFound) {
                dx += static_cast<int>(curand_uniform(&states[idx]) * 3) - 1;
                dy += static_cast<int>(curand_uniform(&states[idx]) * 3) - 1;
            } else {
                dx = static_cast<int>(curand_uniform(&states[idx]) * 3) - 1;
                dy = static_cast<int>(curand_uniform(&states[idx]) * 3) - 1;
            }
        }

        int newX = ant->x + dx;
        int newY = ant->y + dy;

        if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
            ant->x = newX;
            ant->y = newY;

            if (!ant->hasFood && environment[newY * width + newX] == 1) {
                atomicExch(&environment[newY * width + newX], 0);
                ant->hasFood = true;
            }

            else if (ant->hasFood && environment[newY * width + newX] == 3) {
                ant->hasFood = false;
            }
        }
    }
}

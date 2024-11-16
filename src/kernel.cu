//
// Created by Andrea Menci on 15/11/2024.
//

/*
Each cell can contain:
0: Empty
1: Food
3: Nest
*/

#include "kernel.cuh"

// Function to initialize the environment grid
void initializeEnvironment(int* environment, int width, int height, int numFoodSources) {
    // Set all cells to empty
    for (int i = 0; i < width * height; i++) {
        environment[i] = 0;
    }

    // Place the nest in the center of the grid
    environment[width / 2 + height / 2 * width] = 3;

    // Randomly distribute food sources
    srand(time(nullptr));
    for (int i = 0; i < numFoodSources; i++) {
        int x, y;

        // Ensure food is placed in an empty cell
        do {
            x = rand() % width;
            y = rand() % height;
        } while (environment[y * width + x] != 0);

        environment[y * width + x] = 1;
    }
}

// Function to initialize the ants
void initializeAnts(Ant* ants, int numAnts, int width, int height) {
    for (int i = 0; i < numAnts; i++) {
        ants[i].x = width / 2;  // Start ants at the nest position
        ants[i].y = height / 2;
        ants[i].hasFood = false;
    }
}

// CUDA device function to check if two points are within a certain range
__device__ bool isWithinRange(int x1, int y1, int x2, int y2, int range) {
    int dx = x2 - x1;
    int dy = y2 - y1;
    return dx * dx + dy * dy <= range * range;
}

// CUDA kernel to move ants within the environment
__global__ void moveAnts(Ant* ants, int* environment, curandState* states, int numAnts, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numAnts) {
        Ant* ant = &ants[idx];

        // Define the hive's position (center of the grid)
        int hiveX = width / 2;
        int hiveY = height / 2;

        // Ant sensing range, proportional to the grid size
        int senseRange = 0.2 * max(width, height);
        int dx = 0, dy = 0;

        if (ant->hasFood) {
            // If the ant has food, return to the nest
            dx = (hiveX > ant->x) - (hiveX < ant->x);
            dy = (hiveY > ant->y) - (hiveY < ant->y);

            // Add a small random variation to the movement
            dx += static_cast<int>(curand_uniform(&states[idx]) * 3) - 1;
            dy += static_cast<int>(curand_uniform(&states[idx]) * 3) - 1;
        } else {
            // If the ant doesn't have food, search for it
            bool foodFound = false;

            // Scan nearby cells for food
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

            // If food is found, add a small random variation to movement
            if (foodFound) {
                dx += static_cast<int>(curand_uniform(&states[idx]) * 3) - 1;
                dy += static_cast<int>(curand_uniform(&states[idx]) * 3) - 1;
            } else {
                // Move randomly if no food is found
                dx = static_cast<int>(curand_uniform(&states[idx]) * 3) - 1;
                dy = static_cast<int>(curand_uniform(&states[idx]) * 3) - 1;
            }
        }

        // Calculate the new position
        int newX = ant->x + dx;
        int newY = ant->y + dy;

        // Ensure the ant stays within the grid boundaries
        if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
            ant->x = newX;
            ant->y = newY;

            // If the ant picks up food, clear the cell and update its state
            if (!ant->hasFood && environment[newY * width + newX] == 1) {
                atomicExch(&environment[newY * width + newX], 0);
                ant->hasFood = true;
            }

            // If the ant reaches the nest, drop the food
            else if (ant->hasFood && environment[newY * width + newX] == 3) {
                ant->hasFood = false;
            }
        }
    }
}

#include "kernel.cuh"
#include "randomStatesKernel.cuh"
#include <iostream>
#include <fstream>
#include <string>
#include <curand_kernel.h>
#include <cstdlib>
#include <chrono>
#include <thread>

#define NUM_ANTS 4
#define WIDTH 30
#define HEIGHT 30
#define FOOD_SOURCES 3

#define CHECK_CUDA_CALL(call)                                        \
{                                                                    \
    cudaError_t err = call;                                          \
    if (err != cudaSuccess) {                                        \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err)           \
    << " in " << __FILE__ << " at line " << __LINE__ << std::endl;   \
    exit(EXIT_FAILURE);                                              \
    }                                                                \
}

void saveEnvironmentAndAnts(int* environment, Ant* ants, int width, int height, int numAnts, int iteration) {
    int* hostEnvironment = static_cast<int *>(malloc(sizeof(int) * width * height));
    Ant* hostAnts = static_cast<Ant *>(malloc(sizeof(Ant) * numAnts));

    CHECK_CUDA_CALL(cudaMemcpy(hostEnvironment, environment, sizeof(int) * width * height, cudaMemcpyDeviceToHost));
    CHECK_CUDA_CALL(cudaMemcpy(hostAnts, ants, sizeof(Ant) * numAnts, cudaMemcpyDeviceToHost));

    std::ofstream envFile("environment_" + std::to_string(iteration) + ".csv");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            envFile << hostEnvironment[y * width + x];
            if (x < width - 1) envFile << ",";
        }
        envFile << "\n";
    }
    envFile.flush();
    envFile.close();

    std::ofstream antsFile("ants_" + std::to_string(iteration) + ".csv");
    for (int i = 0; i < numAnts; i++) {
        antsFile << hostAnts[i].x << "," << hostAnts[i].y << "," << hostAnts[i].hasFood << "\n";
    }
    antsFile.flush();
    antsFile.close();

    free(hostEnvironment);
    free(hostAnts);
}

int main() {
    int* environment;
    Ant* ants;

    curandState* states;

    CHECK_CUDA_CALL(cudaMalloc(&environment, sizeof(int) * WIDTH * HEIGHT));
    CHECK_CUDA_CALL(cudaMalloc(&ants, sizeof(Ant) * NUM_ANTS));
    CHECK_CUDA_CALL(cudaMalloc(&states, sizeof(curandState) * NUM_ANTS));

    int* hostEnvironment = static_cast<int *>(malloc(sizeof(int) * WIDTH * HEIGHT));
    Ant* hostAnts = static_cast<Ant *>(malloc(sizeof(Ant) * NUM_ANTS));

    initializeEnvironment(hostEnvironment, WIDTH, HEIGHT, FOOD_SOURCES);
    initializeAnts(hostAnts, NUM_ANTS, WIDTH, HEIGHT);

    CHECK_CUDA_CALL(cudaMemcpy(environment, hostEnvironment, sizeof(int) * WIDTH * HEIGHT, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(ants, hostAnts, sizeof(Ant) * NUM_ANTS, cudaMemcpyHostToDevice));

    free(hostEnvironment);
    free(hostAnts);

    dim3 threadsPerBlock(256);
    dim3 numBlocks((NUM_ANTS + threadsPerBlock.x - 1) / threadsPerBlock.x);
    initializeRandomStates<<<numBlocks, threadsPerBlock>>>(states, time(0), NUM_ANTS);
    cudaDeviceSynchronize();

    int iteration = 0;
    bool hasFood = true;

    while (hasFood) {
        moveAnts<<<numBlocks, threadsPerBlock>>>(ants, environment, states, NUM_ANTS, WIDTH, HEIGHT);
        cudaDeviceSynchronize();

        int* checkEnvironment = static_cast<int *>(malloc(sizeof(int) * WIDTH * HEIGHT));
        CHECK_CUDA_CALL(cudaMemcpy(checkEnvironment, environment, sizeof(int) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost));

        hasFood = false;
        for (int i = 0; i < WIDTH * HEIGHT; i++) {
            if (checkEnvironment[i] == 1) {
                hasFood = true;
                break;
            }
        }

        free(checkEnvironment);

        saveEnvironmentAndAnts(environment, ants, WIDTH, HEIGHT, NUM_ANTS, iteration);
        iteration++;
        std::cout << "Iterazione " << iteration << " completata.\n";
    }

    cudaFree(environment);
    cudaFree(ants);
    cudaFree(states);

    return 0;
}

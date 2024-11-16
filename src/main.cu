#include "kernel.cuh"
#include "randomStatesKernel.cuh"
#include <iostream>
#include <fstream>
#include <string>
#include <curand_kernel.h>
#include <cstdlib>
#include <chrono>
#include <thread>

// Define simulation parameters
#define NUM_ANTS 100
#define WIDTH 1000
#define HEIGHT 1000
#define FOOD_SOURCES 100

// Macro for checking CUDA function calls and handling errors
#define CHECK_CUDA_CALL(call)                                        \
{                                                                    \
    cudaError_t err = call;                                          \
    if (err != cudaSuccess) {                                        \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)       \
                  << " in " << __FILE__ << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
}

// Function to save the environment grid and ant positions to CSV files
void saveEnvironmentAndAnts(int* environment, Ant* ants, int width, int height, int numAnts, int iteration) {
    // Allocate memory on the host to store data from the device
    int* hostEnvironment = static_cast<int *>(malloc(sizeof(int) * width * height));
    Ant* hostAnts = static_cast<Ant *>(malloc(sizeof(Ant) * numAnts));

    // Copy data from the device to the host
    CHECK_CUDA_CALL(cudaMemcpy(hostEnvironment, environment, sizeof(int) * width * height, cudaMemcpyDeviceToHost));
    CHECK_CUDA_CALL(cudaMemcpy(hostAnts, ants, sizeof(Ant) * numAnts, cudaMemcpyDeviceToHost));

    // Save the environment grid to a CSV file
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

    // Save the ant positions and states to a CSV file
    std::ofstream antsFile("ants_" + std::to_string(iteration) + ".csv");
    for (int i = 0; i < numAnts; i++) {
        antsFile << hostAnts[i].x << "," << hostAnts[i].y << "," << hostAnts[i].hasFood << "\n";
    }
    antsFile.flush();
    antsFile.close();

    // Free the allocated host memory
    free(hostEnvironment);
    free(hostAnts);
}

int main() {
    // Device pointers for environment, ants, and random states
    int* environment;
    Ant* ants;
    curandState* states;

    // Allocate memory on the device
    CHECK_CUDA_CALL(cudaMalloc(&environment, sizeof(int) * WIDTH * HEIGHT));
    CHECK_CUDA_CALL(cudaMalloc(&ants, sizeof(Ant) * NUM_ANTS));
    CHECK_CUDA_CALL(cudaMalloc(&states, sizeof(curandState) * NUM_ANTS));

    // Allocate memory on the host for initialization
    int* hostEnvironment = static_cast<int *>(malloc(sizeof(int) * WIDTH * HEIGHT));
    Ant* hostAnts = static_cast<Ant *>(malloc(sizeof(Ant) * NUM_ANTS));

    // Initialize the environment and ants on the host
    initializeEnvironment(hostEnvironment, WIDTH, HEIGHT, FOOD_SOURCES);
    initializeAnts(hostAnts, NUM_ANTS, WIDTH, HEIGHT);

    // Copy initialized data from the host to the device
    CHECK_CUDA_CALL(cudaMemcpy(environment, hostEnvironment, sizeof(int) * WIDTH * HEIGHT, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(ants, hostAnts, sizeof(Ant) * NUM_ANTS, cudaMemcpyHostToDevice));

    // Free host memory used for initialization
    free(hostEnvironment);
    free(hostAnts);

    // Configure CUDA kernel execution parameters
    dim3 threadsPerBlock(256);
    dim3 numBlocks((NUM_ANTS + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Initialize random states for ants
    initializeRandomStates<<<numBlocks, threadsPerBlock>>>(states, time(0), NUM_ANTS);
    cudaDeviceSynchronize();

    int iteration = 0;
    bool hasFood = true;

    // Main simulation loop
    while (hasFood) {
        // Move ants according to their state and environment
        moveAnts<<<numBlocks, threadsPerBlock>>>(ants, environment, states, NUM_ANTS, WIDTH, HEIGHT);
        cudaDeviceSynchronize();

        // Check if any food is left in the environment
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

        // Save the current state of the simulation
        saveEnvironmentAndAnts(environment, ants, WIDTH, HEIGHT, NUM_ANTS, iteration);
        iteration++;
        std::cout << "Iteration " << iteration << " completed.\n";
    }

    // Free device memory
    cudaFree(environment);
    cudaFree(ants);
    cudaFree(states);

    return 0;
}

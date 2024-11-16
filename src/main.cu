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
#define FOOD_SOURCES 10

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

// Modify the function to store data in memory
void accumulateEnvironmentAndAnts(int* environment, Ant* ants, int width, int height, int numAnts, std::vector<std::string>& envSnapshots, std::vector<std::string>& antSnapshots) {
    // Temporary allocation on host
    int* hostEnvironment = static_cast<int *>(malloc(sizeof(int) * width * height));
    Ant* hostAnts = static_cast<Ant *>(malloc(sizeof(Ant) * numAnts));

    // Copy data from device to host
    CHECK_CUDA_CALL(cudaMemcpy(hostEnvironment, environment, sizeof(int) * width * height, cudaMemcpyDeviceToHost));
    CHECK_CUDA_CALL(cudaMemcpy(hostAnts, ants, sizeof(Ant) * numAnts, cudaMemcpyDeviceToHost));

    // Generate environment snapshot
    std::ostringstream envStream;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            envStream << hostEnvironment[y * width + x];
            if (x < width - 1) envStream << ",";
        }
        envStream << "\n";
    }
    envSnapshots.push_back(envStream.str());

    // Generate ant snapshot
    std::ostringstream antsStream;
    for (int i = 0; i < numAnts; i++) {
        antsStream << hostAnts[i].x << "," << hostAnts[i].y << "," << hostAnts[i].hasFood << "\n";
    }
    antSnapshots.push_back(antsStream.str());

    // Free temporary host memory
    free(hostEnvironment);
    free(hostAnts);
}

void saveAccumulatedData(const std::vector<std::string>& envSnapshots, const std::vector<std::string>& antSnapshots) {
    // Save environment snapshots
    for (size_t i = 0; i < envSnapshots.size(); ++i) {
        std::ofstream envFile("environment_" + std::to_string(i) + ".csv");
        envFile << envSnapshots[i];
        envFile.close();
    }

    // Save ant snapshots
    for (size_t i = 0; i < antSnapshots.size(); ++i) {
        std::ofstream antsFile("ants_" + std::to_string(i) + ".csv");
        antsFile << antSnapshots[i];
        antsFile.close();
    }
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

    int iteration = 0;
    bool hasFood = true;

    std::vector<std::string> environmentSnapshots;
    std::vector<std::string> antSnapshots;

    // Main simulation loop
    while (hasFood) {
        // Move ants according to their state and environment
        moveAnts<<<numBlocks, threadsPerBlock>>>(ants, environment, states, NUM_ANTS, WIDTH, HEIGHT);

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
        accumulateEnvironmentAndAnts(environment, ants, WIDTH, HEIGHT, NUM_ANTS, environmentSnapshots, antSnapshots);
        iteration++;
        std::cout << "Iteration " << iteration << " completed.\n";
    }

    saveAccumulatedData(environmentSnapshots, antSnapshots);

    // Free device memory
    cudaFree(environment);
    cudaFree(ants);
    cudaFree(states);

    return 0;
}

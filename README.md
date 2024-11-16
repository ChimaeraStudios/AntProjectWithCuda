# Environment Simulation with CUDA

This project implements a simulation of ants interacting with a two-dimensional environment. The computation is optimized using CUDA, while visualization is handled through Python.

## Project Structure

- `main.cu`: Entry point of the program.
- `kernel.cu` and `kernel.cuh`: Definitions and implementations of CUDA functions.
- `randomStatesKernel.cu` and `randomStatesKernel.cuh`: Management of random states.
- `visualize.py`: Python script for result visualization.

## Requirements

1. **CUDA Toolkit** installed on the system.
2. **Python** with the following libraries:
   - `numpy`
   - `matplotlib`

## How to Run

1. Compile the CUDA project:
   ```bash
   nvcc main.cu kernel.cu randomStatesKernel.cu -o simulation
   ```

---

**Group Member:** Andrea Menci  
**GitHub:** https://github.com/Mancee28

# High-Performance 2D Image Gaussian Convolution

This repository contains a implementation of a 2D Gaussian Blur Convolution on PGM images. 
The project focuses on computing the convolution efficiently in order to study parallel scaling,
speedup, and efficiency across different computing paradigms.

## Project Overview

The exact computation of a 2D image convolution is fundamentally limited by the computational cost and memory bandwidth,
scaling as $W \times H \times K^2$, where $W$ and $H$ are the image dimensions and $K$ is the kernel size. 
This project explores the computational limits of shared and distributed memory architectures by progressively 
optimizing the implementation to overcome key HPC bottlenecks, particularly the memory wall. 
The convolution is computed using a dynamically generated Gaussian kernel, and the code evolves 
through multiple levels of optimization, from a baseline serial approach to a fully hybrid MPI + OpenMP domain-decomposition solver.

## Compilation and Execution

The repository contains four main implementations:
* `convolution.c` → serial baseline
* `convolution_omp.c` → OpenMP shared-memory version
* `convolution_mpi.c` → MPI distributed-memory version with halo exchange
* `convolution_hybrid.c` → Hybrid MPI + OpenMP optimized version

### 1. Serial Baseline Version

Baseline implementation executing a standard double-loop convolution.
Used for output validation and to establish the sequential time baseline (highlighting memory scaling limitations).

**Compile**
```bash
gcc -Wall -O3 -o convolution convolution.c -lm
```

**Run**
```bash
./convolution input.pgm output.pgm 5
```

### 2. OpenMP Parallel Version (Shared Memory)

Shared-memory parallel implementation using OpenMP.
This version parallelizes the outer loops over the image rows, sharing the RAM memory efficiently among threads.

**Compile**
```bash
gcc -Wall -O3 -fopenmp -o conv_omp convolution_omp.c -lm
```

**Run**
```bash
./conv_omp input.pgm output.pgm 5
```

### 3. MPI Parallel Version (Distributed Memory)

Distributed-memory implementation using MPI.
MPI distributes the image processing by applying a row-based domain decomposition (`MPI_Scatterv` / `MPI_Gatherv`).
Boundary pixel dependencies are handled via Halo Exchange (ghost rows) using `MPI_Sendrecv`.

**Compile**
```bash
mpicc -Wall -O3 -o conv_mpi convolution_mpi.c -lm
```

**Run**
```bash
mpirun -np 4 ./conv_mpi input.pgm output.pgm 5
```

### 4. Hybrid MPI + OpenMP Version

Fully optimized hybrid implementation:
* **MPI** distributes large chunks of the image over different computing nodes/processes.
* **OpenMP** parallelizes the inner convolution cycles within each local MPI block.

This limits the MPI communication overhead while maximizing core utilization on multi-core nodes.

**Compile**
```bash
mpicc -Wall -O3 -fopenmp -o conv_hybrid convolution_hybrid.c -lm
```

**Run**
```bash
export OMP_NUM_THREADS=2
mpirun -np 2 ./conv_hybrid input.pgm output.pgm 5
```

## Key Features

* Exact application of a 2D Gaussian filter on large-scale PGM images
* Progressive optimization from serial to hybrid formulations
* OpenMP parallelization for shared-memory systems
* MPI row-based domain decomposition with halo exchange for distributed systems
* MPI + OpenMP hybrid scaling for modern cluster architectures
* Analysis of the Memory Wall bottleneck and Hardware Branch Prediction behavior
* Automated BASH benchmarking scripts for performance profiling and CSV data gathering
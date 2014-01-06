cuda_3Dheat
===========

GPU solver for 3-D heat equation using finite differences

Uses __shared__ memory for every slice in the domain and streams
the third dimension in registers. Based on Paulius Micikevicius's
white paper "3D Finite Difference Computation on GPUs using CUDA".


To run:

Run `make` to create executable `bin/cuda_3Dheat.bin`
Executable puts `output.dat` in results/

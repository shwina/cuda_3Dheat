cuda_3Dheat : obj/cuda_3Dheat.o
	nvcc -o bin/cuda_3Dheat.o obj/cuda_2Dheat.o

obj/cuda_2Dheat.o : src/cuda_3Dheat.cu
	nvcc -arch=sm_20 -o obj/cuda_3Dheat.o -I src/include src/cuda_3Dheat.cu


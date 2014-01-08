cuda_3Dheat : obj/cuda_3Dheat.o
	nvcc -o bin/cuda_3Dheat.bin obj/cuda_3Dheat.o

obj/cuda_3Dheat.o : src/cuda_3Dheat.cu src/include/3D_heat_kernel.cu params.yml
	nvcc -arch=sm_20 -o obj/cuda_3Dheat.o -I src/include -c src/cuda_3Dheat.cu


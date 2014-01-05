# include <stdlib.h>
# include <time.h>
# include <init.cu>
# include <3D_heat_kernel.cu>
# include <printmatrix.hpp>
# include <sys/time.h>


int main()
{

    float   L_x = 16,
            L_y = 16,
            L_z = 32;

    int     N_x = 64,
            N_y = 64,
            N_z = 32;

    float   dx = L_x/N_x,
            dy = L_y/N_y,
            dz = L_z/N_z,
            alpha = 0.1, 
            dt = 0.1;

    int     nsteps = 100;


    float *temp1_h, *temp1_d, *temp2_d, *temp_tmp;

    // Allocate memory and intialise temperatures in host    
    int ary_size = N_x * N_y * N_z * sizeof(float);
    temp1_h = (float *)malloc(ary_size);
    init_temp(temp1_h, N_x, N_y, N_z);

    // Allocate memory on device and copy from host
    cudaMalloc((void**)&temp1_d, ary_size);
    cudaMalloc((void**)&temp2_d, ary_size);

    cudaMemcpy((void *)temp1_d, (void *)temp1_h, ary_size,
                cudaMemcpyHostToDevice);

    cudaMemcpy((void *)temp2_d, (void *)temp1_h, ary_size,
                cudaMemcpyHostToDevice);

    // Launch configuration:
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(N_x/16, N_y/16, 1);
    
    // Compute on device
    
    struct timeval t1, t2;
    double elapsedtime;

    timeval(&t1, NULL);
    for (int i=0; i<nsteps; i++){
        temperature_update16x16<<<dimGrid, dimBlock>>>(temp1_d, temp2_d, alpha,
                                                        dt, N_x, N_y, N_z,
                                                        dx, dy, dz);
        cudaThreadSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
                // print the CUDA error message and exit
                printf("CUDA error: %s\n", cudaGetErrorString(error));
                exit(-1);
        }
        temp_tmp = temp1_d;
        temp1_d  = temp2_d;
        temp2_d  = temp_tmp;

    }
    timeval(&t2, NULL);
    elapsedtime = (t2.tv_sec - t1.tv_sec);
    printf("Temperature update loop took: %f\n seconds", elapsedtime);

    // Copy from device to host
    cudaThreadSynchronize();
    cudaMemcpy((void*) temp1_h, (void*) temp2_d, ary_size,
                cudaMemcpyDeviceToHost);

    write_to_file(temp1_h, N_x*N_y*N_z,"output.dat");

    return 0;

}

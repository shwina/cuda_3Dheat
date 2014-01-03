# include <stdlib.h>
# include <time.h>
# include <init.cu>
# include <3D_heat_kernel.cu>

int main()
{

    float   L_x = 64,
            L_y = 64,
            L_z = 32;

    int     N_x = 64,
            N_y = 64,
            N_z = 32;

    float   dx = L_x/N_x,
            dy = L_y/N_y,
            dz = L_z/N_z,
            alpha = 0.1, 
            dt = 1;

    int     nsteps = 10;


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
    for (int i=0; i<nsteps; i++){
        temperature_update<<<blockDim, gridDim>>>(temp1_d, temp2_d, alpha,
                                                    dt, N_x, N_y, N_z,
                                                    dx, dy, dz);
        temp_tmp = temp1_d;
        temp1_d  = temp2_d;
        temp2_d  = temp_tmp;
    }

    // Copy from device to host
    cudaThreadSynchronize();
    cudaMemcpy((void*) temp1_h, (void*) temp1_d, ary_size,
                cudaMemcpyDeviceToHost);

    // Write to file

    return 0;

}
# include <stdlib.h>
# include <time.h>
# include <init.cu>
# include <3D_heat_kernel.cu>
# include <printmatrix.hpp>
# include <sys/time.h>
# include <read_params.hpp>

int main()
{
    // Read grid and simulation parameters:

    YAML::Node doc = YAML::LoadFile("../src/include/params.yml");
    Grid g = doc["grid_params"].as<Grid>();
    Sim s  = doc["sim_params"].as<Sim>();

    float dx = g.L_x/g.N_x;
    float dy = g.L_y/g.N_y;
    float dz = g.L_z/g.N_z;

    float *temp1_h, *temp1_d, *temp2_d, *temp_tmp;

    // Allocate memory and intialise temperatures in host    
    int ary_size = g.N_x * g.N_y * g.N_z * sizeof(float);   
    temp1_h = (float *)malloc(ary_size);
    init_temp(temp1_h, g.N_x, g.N_y, g.N_z);

    // Allocate memory on device and copy from host
    cudaMalloc((void**)&temp1_d, ary_size);
    cudaMalloc((void**)&temp2_d, ary_size);
    cudaMemcpy((void *)temp1_d, (void *)temp1_h, ary_size,
                cudaMemcpyHostToDevice);
    cudaMemcpy((void *)temp2_d, (void *)temp1_h, ary_size,
                cudaMemcpyHostToDevice);

    // Launch configuration:
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(g.N_x/16, g.N_y/16, 1);
    
    // Compute on device
    
    struct timeval t1, t2;
    double elapsedtime;

    gettimeofday(&t1, NULL);
    for (int i=0; i<s.nsteps; i++){
        temperature_update16x16<<<dimGrid, dimBlock>>>(temp1_d, temp2_d, s.alpha,
                                                        s.dt, g.N_x, g.N_y, g.N_z,
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
    gettimeofday(&t2, NULL);
    elapsedtime = (t2.tv_usec - t1.tv_usec);
    printf("Temperature update loop took: %f\n microseconds", elapsedtime);

    // Copy from device to host
    cudaThreadSynchronize();
    cudaMemcpy((void*) temp1_h, (void*) temp2_d, ary_size,
                cudaMemcpyDeviceToHost);

    write_to_file(temp1_h, g.N_x*g.N_y*g.N_z,"output.dat");

    return 0;

}

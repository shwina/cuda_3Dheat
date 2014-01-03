# include <stdio.h>


__global__ void temperature_update(float* temp1_d, float* temp2_d, float alpha, 
                    float dt, 
                    const int N_x, const int N_y, const int N_z,
                    const float dx, const float dy, const float dz){


#define BDIMX   16
#define BDIMY   16

    // Load a slice into shared memory:
    __shared__ float slice[BDIMY + 2][BDIMX + 2];


    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int in_idx = iy*N_x + ix;  
    int out_idx = 0;
    int stride = N_x*N_y;


    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    // The thread has a "personal" copy of the variable
    // in front of and behind the slice:

    float   current = temp1_d[in_idx],
            infront = temp1_d[in_idx + stride],
            behind  = temp1_d[in_idx + 2*stride];

    // Since the "radius" is really just 1:
    for (int i = 1; i < N_z - 1; i++){

        // Advance the slice:
        behind  = current;
        current = infront;
        infront = in_idx;

        in_idx += stride;
        out_idx += stride;

        __syncthreads();

        // Update the data slice in shared mem:
        if (threadIdx.y<1){ // Halo above/below
            slice[threadIdx.y][tx]              = temp1_d[out_idx-N_x];
            slice[threadIdx.y+BDIMY+1][tx] = temp1_d[out_idx+BDIMY*N_x];
        }

        if (threadIdx.x<1){ // Halo left/right
            slice[ty][threadIdx.x]              = temp1_d[out_idx-1];
            slice[ty][threadIdx.x+BDIMX+1] = temp1_d[out_idx+BDIMX];
        }

        slice[ty][tx] = current;
        __syncthreads();

        // Update temperature at output point:

        temp2_d[out_idx] =  current + 
                            (alpha*dt)*((slice[ty-1][tx] - 
                            2*current + slice[ty+1][tx])/(dx*dx) + 
                            (slice[ty][tx-1] - 
                            2*current + slice[ty][tx+1])/(dy*dy) +
                            (infront - 2*current + behind)/(dz*dz));

    }

}
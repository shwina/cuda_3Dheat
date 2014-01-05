
/*
    Kernel to do temperature update in explicit finite difference
    solution to 3-D heat equation. Works for a block size of 16 x 16.
    Make copies for other block sizes. Can be easily extended to 
    arbitrary sized stencils.
*/


# include <stdio.h>
# include <cuda.h>

__global__ void temperature_update16x16(float* temp1_d, float* temp2_d, float alpha, 
                    float dt, 
                    const int N_x, const int N_y, const int N_z,
                    const float dx, const float dy, const float dz){


#define BDIMX 16
#define BDIMY 16

__shared__ float slice[BDIMX+2][BDIMY+2];

int ix = blockIdx.x*blockDim.x + threadIdx.x;
int iy = blockIdx.y*blockDim.y + threadIdx.y;

int tx = threadIdx.x + 1;
int ty = threadIdx.y + 1;

int stride = N_x*N_y;
int i2d = iy*N_x + ix;
int o2d = 0;
bool compute_if = ix > 0 && ix < (N_x-1) && iy > 0 && iy < (N_y-1);


float behind;
float current = temp1_d[i2d]; o2d = i2d; i2d += stride;
float infront = temp1_d[i2d]; i2d += stride;

for(int i=1; i<N_z-1; i++){

    // These go in registers:
    behind = current;
    current= infront;
    infront= temp1_d[i2d];

    i2d += stride;
    o2d += stride;
    __syncthreads();

    // Shared memory

    if (compute_if){
        if(threadIdx.x == 0){ // Halo left
            slice[ty][tx-1]     =   temp1_d[o2d - 1];
        }

        if(threadIdx.x == BDIMX-1){ // Halo right
            slice[ty][tx+1]     =   temp1_d[o2d + 1];
        }

        if(threadIdx.y == 0){ // Halo bottom
            slice[ty-1][tx]     =   temp1_d[o2d - N_x];
        }

        if(threadIdx.y == BDIMY-1){ // Halo top
            slice[ty+1][tx]     =   temp1_d[o2d + N_x];
        }
    }

    __syncthreads();

    slice[ty][tx] = current;

    __syncthreads();

    if (compute_if){
        
        temp2_d[o2d]  = current + (alpha*dt)*(
                        (slice[ty][tx-1] - 2*slice[ty][tx]
                        +slice[ty][tx+1])/(dx*dx) +
                        (slice[ty-1][tx] - 2*slice[ty][tx]
                        +slice[ty+1][tx])/(dy*dy) +
                        (behind - 2*current + infront)/(dz*dz));

    }

    __syncthreads();

}

}


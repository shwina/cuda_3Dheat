
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
bool compute_if = ix > 0 && ix < N_x-1 && iy > 0 && iy < N_y-1;


for(int i=1; i<N_z-1; i++){

    i2d += stride;
    // These go in registers:
    float behind = temp1_d[i2d-stride];
    float current= temp1_d[i2d];
    float infront= temp1_d[i2d+stride];

    // Shared memory
    slice[threadIdx.y+1][threadIdx.x+1] = temp1_d[i2d];

    if (compute_if){
         if(threadIdx.x == 0){ // Halo left
            slice[ty][0]     =   temp1_d[i2d - 1];
        }

        if(threadIdx.x == BDIMX-1){ // Halo right
            slice[ty][BDIMX+1] = temp1_d[i2d + 1];
        }

        if(threadIdx.y == 0){ // Halo bottom
            slice[0][tx]     =   temp1_d[i2d - BDIMX];
        }

        if(threadIdx.y == BDIMY-1){ // Halo top
            slice[BDIMY+1][tx] = temp1_d[i2d + BDIMX];
        }
    }

    __syncthreads();

    if (compute_if){
        
        temp2_d[i2d]  = current + (alpha*dt)*(
                        (slice[ty-1][tx] - 2*slice[ty][tx]
                        +slice[ty+1][tx])/(dx*dx) +
                        (slice[ty][tx-1] - 2*slice[ty][tx]
                        +slice[ty][tx+1])/(dy*dy) +
                        (behind - 2*current + infront)/(dz*dz));

    }

    __syncthreads();

}

}


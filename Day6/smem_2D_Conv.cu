#include <iostream>
#include <cuda_runtime.h>

#define ROWS 32
#define COLS 32
#define BLOCK_SIZE 8
#define KERNEL_SIZE 5
#define RADIUS KERNEL_SIZE/2

using namespace std;

__global__
void tiled_2D_Conv(const int *A, int *B, const int *k) {

    __shared__ int smem[BLOCK_SIZE+2*RADIUS][BLOCK_SIZE+2*RADIUS];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    auto idx = [&](int r, int c) {
        return r * COLS + c;
    };

    // center copy, the actual block

    if (row < ROWS && col < COLS) {
        smem[ty+RADIUS][tx+RADIUS] = A[idx(row, col)];
    }
    
    // left and right halo copy

    if (tx < RADIUS) {
        int left = col - RADIUS;
        int right = col + BLOCK_SIZE;

        smem[ty+RADIUS][tx] = (left >= 0 && row < ROWS)? A[idx(row, left)] : 0;
        smem[ty+RADIUS][tx+BLOCK_SIZE] =  (right < COLS && row < ROWS)? A[idx(row, right)] : 0;
    }

    // top and bottom halo copy

    if (ty < RADIUS) {
        int top = row - RADIUS;
        int bottom = row + BLOCK_SIZE;

        smem[ty][tx+RADIUS] = (top >= 0 && col < COLS) ? A[idx(top, col)] : 0;
        smem[ty+BLOCK_SIZE+RADIUS][tx+RADIUS] = (bottom < ROWS && col < COLS) ? A[idx(bottom, col)] : 0;
    }

    // corner copy
    
    if (tx < RADIUS && ty < RADIUS) {

        int top = row-RADIUS;
        int bottom = row+BLOCK_SIZE;

        int left = col-RADIUS;
        int right = col+BLOCK_SIZE;
        
        smem[ty][tx] = (top >= 0 && left >= 0) ? A[idx(top, left)] : 0;
        smem[ty][tx+BLOCK_SIZE+RADIUS] = (top >= 0 && right < COLS) ? A[idx(top, right)] : 0;
        smem[ty+BLOCK_SIZE+RADIUS][tx] = (bottom < ROWS && left >= 0) ? A[idx(bottom, left)] : 0;
        smem[ty+BLOCK_SIZE+RADIUS][tx+BLOCK_SIZE+RADIUS] = (bottom < ROWS && right < COLS) ? A[idx(bottom, right)] : 0;
    }

    __syncthreads();


    if (row < ROWS && col < COLS) {
        int total = 0;

        for (int ky=-RADIUS; ky<=RADIUS; ky++) {
            for (int kx=-RADIUS; kx<=RADIUS; kx++) {
                int r = ty + RADIUS + ky;
                int c = tx + RADIUS + kx;

                int j = (ky + RADIUS) * KERNEL_SIZE + (kx + RADIUS);
                total += smem[r][c] * k[j];
            }
        }

        B[idx(row, col)] = total;
    }

}

int main() {

    int *A = new int[ROWS*COLS];
    int *B = new int[ROWS*COLS];
    int *kernel = new int[KERNEL_SIZE*KERNEL_SIZE];

    int *dA, *dB, *k;

    for (size_t i = 0; i<ROWS*COLS; i++) {
        // A[i] = rand() % 100;
        A[i] = 1;
    }

    for (size_t i = 0; i<KERNEL_SIZE*KERNEL_SIZE; i++) {
        kernel[i] = 5;
    }
    
    cudaMalloc(&dA, sizeof(int)*ROWS*COLS);
    cudaMalloc(&dB, sizeof(int)*ROWS*COLS);
    cudaMalloc(&k, sizeof(int)*KERNEL_SIZE*KERNEL_SIZE);

    cudaMemcpy(dA, A, sizeof(int)*ROWS*COLS, cudaMemcpyHostToDevice);
    cudaMemcpy(k, kernel, sizeof(int)*KERNEL_SIZE*KERNEL_SIZE, cudaMemcpyHostToDevice);
    
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((COLS+BLOCK_SIZE-1)/BLOCK_SIZE, (ROWS+BLOCK_SIZE-1)/BLOCK_SIZE);

    tiled_2D_Conv<<<grid_dim, block_dim>>>(dA, dB, k);

    cudaMemcpy(B, dB, sizeof(int)*ROWS*COLS, cudaMemcpyDeviceToHost);

    for (int i=0; i < 5; i++) {
        cout << B[i] << ", ";
    }
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(k);

    delete[] A;
    delete[] B;
    delete[] kernel;



    return 0;
}
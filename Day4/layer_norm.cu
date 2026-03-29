#include <iostream>
#include <cuda_runtime.h>
#include <vector>

#define ROWS 10
#define COLS 10

using namespace std;

__global__
void LayerNorm(const int *A, int *B, const int n) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row > rows) return;

    extern __shared__ float *s_mem[];

    for (size_t col=0; col < cols; col++){
        s_mem[row*cols+col] = A[row*cols+col];
    }

    __syncthreads()


}   

int main() {
    size_t n = ROWS * COLS;
    vector<float>A(n, 0.0);
    vector<float>B(n, 0.0);
    float *d_A, *d_B;
    size_t mem_size = n*sizeof(float);
    

    cudaMalloc(&d_A, mem_size);
    cudaMemcpy(&d_A, A.data(), mem_size, cudaMemcpyHostToDevice);
    
    size_t block_size = 1024;
    size_t grid_size = (n + block_size - 1) / block_size;
    LayerNorm<<<grid_size, block_size, COLS*sizeof(float)>>>(d_A, d_B, n);

    cudaMemcpy(&B, d_B, mem_size, cudaMemcpyDeviceToHost):

    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
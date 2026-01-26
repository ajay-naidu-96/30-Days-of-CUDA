#include <cuda_runtime.h>
#include <iostream>
#include "utils.h"

__global__ 
void mat_mul(const float *A, const float *B, float *C, const int N) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    size_t j = blockDim.y * blockIdx.y + threadIdx.y;

    if ((i >= N) || (j >= N)) {
        return ;
    }

    float total = 0.0;

    for (size_t k=0; k < N; k++) {
        total += A[i*N+k] * B[k*N+j];
    }

    C[i*N+j] = total;
}

int main() {
    size_t N = 10;
    size_t mem_size = N*N;

    float A[mem_size], B[mem_size], C[mem_size];
    float *a, *b, *c;

    float num = 0;

    for (size_t r=0; r<N; r++) {
        for (size_t c=0; c<N; c++) {
            A[r*N+c] = num;
            B[r*N+c] = num+2;
            num++;
        }
    }

    cudaMalloc(&a, mem_size*sizeof(float));
    cudaMalloc(&b, mem_size*sizeof(float));
    cudaMalloc(&c, mem_size*sizeof(float));

    cudaMemcpy(a, A, mem_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b, B, mem_size*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_size(16, 16);
    dim3 grid_size((block_size.x+N-1)/block_size.x, (block_size.y+N-1)/block_size.y);

    mat_mul<<<grid_size, block_size>>>(a, b, c, N);

    cudaMemcpy(C, c, mem_size*sizeof(float), cudaMemcpyDeviceToHost);
    
    print(A, N, N, "A");
    print(B, N, N, "B");
    print(C, N, N, "C");

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
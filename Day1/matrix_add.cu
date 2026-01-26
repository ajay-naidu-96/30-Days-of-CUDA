#include <iostream>
#include "utils.h"
#include <cuda_runtime.h>

// mismatch between array types and kernel called, int num and float mul / add 

__global__ void matrix_add(const float *A, const float *B, float *C, int N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i >= N) ||(j >= N)) {
        return;
    }

    C[i*N+j] = A[i*N+j] + B[i*N+j];
}

// awfully mediocre versions depending on the machine of execution

__global__ void matrix_add_row_major(const float *A, const float *B, float *C, int N) {

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) {
        return ;
    }
    else {
        for (size_t j=0; j < N; j++) {
            C[i*N+j] = A[i*N+j] + B[i*N+j];
        }
    }

}

__global__ void matrix_add_col_major(const float *A, const float *B, float *C, int N) {

    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (j >= N) {
        return;
    }
    else {
        for (size_t i = 0; i < N; i++) {
            C[i*N+j] = A[i*N+j] + B[i*N+j];
        }
    }
    
}


int main() {
    size_t N = 10;

    float A[N*N];
    float B[N*N];
    float C[N*N];

    float *a, *b, *c;

    int num = 0;

    for (size_t r=0; r<N; r++) {
        for (size_t c=0; c<N; c++) {
            A[r*N+c] = num;
            B[r*N+c] = num+2;
            num += 1;
        }
    }

    size_t mem_size = N*N;

    cudaMalloc(&a, sizeof(float)*mem_size);
    cudaMalloc(&b, sizeof(float)*mem_size);
    cudaMalloc(&c, sizeof(float)*mem_size);

    cudaMemcpy(a, A, mem_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b, B, mem_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c, C, mem_size*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_size(16, 16);
    dim3 grid_size(
        N+block_size.x-1/block_size.x, 
        N+block_size.y-1/block_size.y
    );

    matrix_add<<<grid_size, block_size>>>(a, b, c, N);

    print(A, N, N, "A");
    print(B, N, N, "B");

    cudaMemcpy(C, c, mem_size*sizeof(float), cudaMemcpyDeviceToHost);

    print(C, N, N, "C");

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;

}
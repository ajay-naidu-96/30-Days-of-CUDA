#include <iostream>
#include "utils.h"

using namespace std;

__global__ void vec_add(const float *A, const float *B, float *C, size_t N) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }

}

int main() {
    
    size_t N = 10;

    float A[N];
    float B[N];
    float C[N];

    float *a, *b, *c;

    for (size_t i=0; i<N; i++) {
        A[i] = i;
        B[i] = i+1;
    }
    
    size_t mem_size = N*sizeof(float);

    cudaMalloc(&a, mem_size);
    cudaMalloc(&b, mem_size);
    cudaMalloc(&c, mem_size);

    size_t block_size = 256;
    size_t grid_size = ceil((N+block_size-1)/block_size);

    cudaMemcpy(a, A, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b, B, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(c, C, mem_size, cudaMemcpyHostToDevice);

    vec_add<<<grid_size, block_size>>>(a, b, c, N);

    cudaMemcpy(C, c, mem_size, cudaMemcpyDeviceToHost);

    print(A, N, 1, "A");
    print(B, N, 1, "B");
    print(C, N, 1, "C");
    
    // cout << "Array A: " << endl;
    // cout << "[";

    // for (size_t i=0; i < N; i++) {
    //     cout << A[i] << ", ";
    // }

    // cout << "]" << endl;
    
    // cout << "Array B: " << endl;
    // cout << "[";

    // for (size_t i=0; i < N; i++) {
    //     cout << B[i] << ", ";
    // }

    // cout << "]" << endl;

    // cout << "Array C: " << endl;
    // cout << "[";

    // for (size_t i=0; i < N; i++) {
    //     cout << C[i] << ", ";
    // }

    // cout << "]" << endl;

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

}
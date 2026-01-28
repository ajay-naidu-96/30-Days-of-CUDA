#include<iostream>
#include <vector>
#include <cuda_runtime.h>
using namespace std;

#define BLOCK_SIZE 256
const size_t N = 100000000;

__global__
void block_linear_scan(const int *A, int *P, int *B, const int N) {

    size_t bid = blockIdx.x;
    size_t start = bid * blockDim.x;

    if (threadIdx.x == 0) {
        int run_sum = 0;
        for (size_t i = 0; ((i < blockDim.x) && (start+i < N)); i++) {
            run_sum += A[start+i];
            P[start+i] = run_sum;
        }

        if (B) {
            B[bid] = run_sum;
        }
    }
}

__global__
void add_blocks(int *B, const int n_blocks) {

    if (threadIdx.x == 0) {
        for (size_t i=1; i < n_blocks; i++) {
            B[i] += B[i-1];
        }
    }

}

__global__
void add_offsets(int *P, int *B, const int N) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    if ((gid < N) && (blockIdx.x > 0)) {
        P[gid] += B[blockIdx.x-1];
    }
}

int main() {
    vector<int> A(N, 1);
    int *d_A, *d_P;
    int *d_B;

    size_t num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t mem_size = N*sizeof(int);

    cudaMalloc(&d_A, mem_size);
    cudaMalloc(&d_P, mem_size);
    cudaMalloc(&d_B, num_blocks * sizeof(int));
    
    cudaMemset(&d_P, 0, mem_size);
    cudaMemset(&d_B, 0, num_blocks*sizeof(int));

    cudaMemcpy(d_A, A.data(), mem_size, cudaMemcpyHostToDevice);

    block_linear_scan<<<num_blocks, BLOCK_SIZE>>> (d_A, d_P, d_B, N);
    add_blocks<<<1, 1>>> (d_B, num_blocks);
    add_offsets<<<num_blocks, BLOCK_SIZE>>> (d_P, d_B, N);
    
    int last_val = 0;

    cudaMemcpy(&last_val, &d_P[N-1], sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_P);
    cudaFree(d_B);

    cout << "Last Val: " << last_val << endl;

    return 0;
    
}
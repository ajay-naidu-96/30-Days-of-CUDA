#include "cuda_runtime.h"
#include <iostream>

using namespace std;

__device__
void co_rank(const int *A, const int *B, const int K, const int N, const int M, int *i_o, int *j_o) {

    int low = max(0, K-M);
    int high = min(K, N);

    while (low <= high) {
        int i = (low+high) / 2;
        int j = K - i;
        
        // bounds check for j because j depends on i and k
        if (j < 0) {
            high = i-1;
            continue;
        }
        else if (j > M) {
            low = i+1;
            continue;
        }

        if (i > 0 && j < M && A[i-1] > B[j]) {
            high = i-1;
        }
        else if (j > 0 && i < N && B[j-1] > A[i]) {
            low = i+1;
        }
        else {
            *i_o = i;
            *j_o = j;
            return;
        }


    }


}

__global__
void merge_sort(const int *A, const int *B, int *C, const int N, const int M) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= (N+M)) {
        return;
    }
    
    int i, j;

    co_rank(A, B, tid, N, M, &i, &j);

    if (j > M || (i < N and A[i] <= B[j])) {
        C[tid] = A[i];
    }   
    else {
        C[tid] = B[j];
    }

}

int main() {

    size_t N = 100;
    size_t M = 100;
    size_t block_size = 1024;

    int *A = new int[N];
    int *B = new int[M];

    int *C = new int[N+M]();

    for (size_t i=0; i<N; i++) {
        A[i] = 2 * i;
    }

    for (size_t i=0; i<M; i++) {
        B[i] = 2 * i + 1;
    }

    int *dA, *dB, *dC;

    cudaMalloc(&dA, sizeof(int)*N);
    cudaMalloc(&dB, sizeof(int)*M);
    cudaMalloc(&dC, sizeof(int)*(N+M));

    cudaMemcpy(dA, A, sizeof(int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(int)*N, cudaMemcpyHostToDevice);

    dim3 block(block_size);
    dim3 grid((N+M+block.x-1)/block.x);

    merge_sort<<<grid, block>>>(dA, dB, dC, N, M);

    cudaMemcpy(C, dC, sizeof(int)*(N+M), cudaMemcpyDeviceToHost);

    for (size_t i=0; i<(N+M); i++) {
        cout << C[i] << ", ";
    }

    delete A;
    delete B;
    delete C;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
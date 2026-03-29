#include <iostream>
#include <vector>

#define N 100000000
#define BLOCK_SIZE 1024
#define K 5

using namespace std;

__global__
void Tile1DConv(const int *A, const int *kernel, int *C) {
    extern __shared__ int s_A[];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (gid < N) {
        s_A[tid+K/2] = A[gid]; 
    }

    if (tid < K/2) {
        int l = blockIdx.x * blockDim.x + tid - K/2;
        s_A[tid] = (l >= 0) ? A[l] : 0;

        int r = blockIdx.x * blockDim.x + blockDim.x + tid;
        s_A[BLOCK_SIZE + K/2 + tid] = (r < N) ? A[r] : 0;
    }

    __syncthreads();

    int total = 0;

    for (int i=-(K/2); i <= (K/2); i++) {
        if (gid+i >= 0 and gid+i < N) {
            total += s_A[tid+i+K/2] * kernel[i+K/2];
        }
    }

    if (gid < N)
        C[gid] = total;

}   

int main() {
    vector<int> A(N, 0);
    vector<int> C(N, 0);
    int *d_A, *d_C, *d_K;

    int kernel[K];

    for (size_t i=0; i<K; i++) {
        kernel[i] = rand() % 100;
    }

    for (size_t i=0; i<N; i++) {
        A[i] = rand() % 699; 
    }

    cudaMalloc(&d_A, N*sizeof(int));
    cudaMemcpy(d_A, A.data(), N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_K, K*sizeof(int));
    cudaMemcpy(d_K, kernel, K*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_C, N*sizeof(int));

    size_t num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    Tile1DConv<<<num_blocks, BLOCK_SIZE, (BLOCK_SIZE+K-1)*sizeof(int)>>>(d_A, d_K, d_C);
    
    cudaMemcpy(C.data(), d_C, 5*sizeof(int), cudaMemcpyDeviceToHost);

    cout << "A: [";
    for (size_t i=0; i < K; i++) {
        cout << A[i] << ", ";
    }
    cout << "]" << endl;;

    cout << "kernel: [";
    for (size_t i=0; i < K; i++) {
        cout << kernel[i] << ", ";
    }
    cout << "]" << endl;;

    cout << "C: [";
    for (size_t i =0; i < K; i++) {
        cout << C[i] << ", ";
    }
    cout << "]";



    return 0;
}

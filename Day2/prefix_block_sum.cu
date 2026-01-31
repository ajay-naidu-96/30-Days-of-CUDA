#include <iostream>
#include <vector>
#define BLOCK_SIZE 256

const int N = 100000000;
using namespace std;

__global__
void block_linear_scan(int *A, int *B, int num_blocks) {
    size_t tid = threadIdx.x;
    size_t gid = blockIdx.x * blockDim.x;

    if (tid == 0) {
        int run_sum = 0;
        for (size_t i=0; (gid+i < N)  && (i < blockDim.x); i++) {
            run_sum += A[gid+i];
            A[gid+i] = run_sum;
        }
        if (B) {
            B[blockIdx.x] = run_sum;
        }
    }
}

__global__
void add_blocks(int *B, int num_blocks) {
    size_t tid = threadIdx.x;

    if (tid == 0) {
        for (size_t i = 1; i < num_blocks; i++) {
            B[i] += B[i-1];
        }
    }
}


__global__
void add_offsets(int *A, int *B, int num_blocks) {

    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if ((blockIdx.x > 0) && (gid < N)) {
        A[gid] += B[blockIdx.x-1];
    } 

}


int main() {
    vector<int> A(N, 1);
    int *d_A, *d_B;
    int last_val;

    size_t num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMalloc(&d_A, sizeof(int)*N);
    cudaMalloc(&d_B, sizeof(int)*num_blocks);
    cudaMemset(d_B, 0, sizeof(int)*num_blocks);

    cudaMemcpy(d_A, A.data(), N*sizeof(int), cudaMemcpyHostToDevice);

    block_linear_scan<<<num_blocks, BLOCK_SIZE>>>(d_A, d_B, num_blocks);
    add_blocks<<<1,1>>>(d_B, num_blocks);
    add_offsets<<<num_blocks, BLOCK_SIZE>>>(d_A, d_B, num_blocks);

    cudaMemcpy(&last_val, d_A + N-1, sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Last Val: " << last_val << endl;

    cudaFree(d_A);
    cudaFree(d_B);
    
    return 0;
}
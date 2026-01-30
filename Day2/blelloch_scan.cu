#include <iostream>
#include <vector>
#define BLOCK_SIZE 256

const int N = 100000000;
using namespace std;

__global__
void scan(int *A, int *B, int num_blocks) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    __shared__ int temp[BLOCK_SIZE];

    temp[tid] = gid < N ? A[gid] : 0;
    __syncthreads();

    // int val = temp[tid];
    
    // Reduction
    for (int offset=1; offset < BLOCK_SIZE; offset <<= 1) {
        int idx = (tid + 1) * 2 * offset - 1;

        if (idx < BLOCK_SIZE) {
            temp[idx] += temp[idx-offset];
        }

        __syncthreads();
    }


    if (B && tid == 0) {
        B[blockIdx.x] = temp[BLOCK_SIZE-1];
    }

    if (tid == 0) {
        temp[BLOCK_SIZE-1] = 0;
    }
    __syncthreads();
    
    for(int offset=BLOCK_SIZE>>1; offset > 0; offset >>= 1) {
        int idx = (tid + 1) * 2 * offset - 1;
        
        if (idx < BLOCK_SIZE) {
            int t = temp[idx-offset];
            temp[idx-offset] = temp[idx];
            temp[idx] += t;
        }

        __syncthreads();
    }

    if (gid < N) {
        A[gid] += temp[tid];
    }

}

__global__
void add_blocks(int *B, int num_blocks) {
    
    int tid = threadIdx.x;

    if (tid == 0) {
        for (int i = 1; i < num_blocks; i++) {
            B[i] += B[i-1];
        }
    }
}

__global__
void add_offsets(int *A, int *B, int num_blocks) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if ((gid < N) && (blockIdx.x > 0)) {
        A[gid] += B[blockIdx.x-1];
    }
}

int main() {
    vector<int>a(N, 1);
    int *d_A, *d_B;
    int last;

    cudaMalloc(&d_A, N*sizeof(int));
    cudaMemcpy(d_A, a.data(), N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_B, BLOCK_SIZE*sizeof(int));
    cudaMemset(&d_B, 0, BLOCK_SIZE*sizeof(int));

    size_t num_blocks = (N+BLOCK_SIZE-1) / BLOCK_SIZE;

    scan<<<num_blocks, BLOCK_SIZE>>>(d_A, d_B, num_blocks);
    add_blocks<<<1, 1>>>(d_B, num_blocks);
    add_offsets<<<num_blocks, BLOCK_SIZE>>>(d_A, d_B, num_blocks);

    cudaMemcpy(&last, d_A+N-1, sizeof(int), cudaMemcpyDeviceToHost);

    cout << "last: " << last << endl;

    cudaFree(d_B);
    cudaFree(d_A);
    
    return 0;
}


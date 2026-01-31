#include <iostream>
#define N 100000000
#define BLOCK_SIZE 1024
#include <vector>

using namespace std;

__global__
void scan_linear_block(int *A, int *B, size_t n) {
    size_t tid = threadIdx.x;
    size_t start = blockDim.x * blockIdx.x;
    
    if (tid == 0) {
        for (size_t i = 1; (i < BLOCK_SIZE) && (start+i < n); i++) {
            size_t gid = start + i;
            A[gid] += A[gid-1];
        }
        
        if (B) {
            B[blockIdx.x] = A[start+BLOCK_SIZE-1];
        }

    }
}

__global__
void add_offsets(int *A, int *B, size_t n) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.x > 0) {
        A[gid] += B[blockIdx.x-1];  
    }
}



void scan_block_recursive(int *A, size_t n) {

    if (n <= 1) {
        return;
    }
    
    if (n < BLOCK_SIZE) {
        scan_linear_block<<<1, BLOCK_SIZE>>>(A, nullptr, n);
        return ;
    }

    int *d_B;
    size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMalloc(&d_B, sizeof(int)*num_blocks);    

    scan_linear_block<<<num_blocks, BLOCK_SIZE>>>(A, d_B, n);
    scan_block_recursive(d_B, num_blocks);
    add_offsets<<<num_blocks, BLOCK_SIZE>>>(A, d_B, n);

    cudaFree(d_B);
}


int main() {

    vector<int> A(N, 1);
    int *d_A;
    int last_val;

    cudaMalloc(&d_A, N*sizeof(int));
    cudaMemcpy(d_A, A.data(), N*sizeof(int), cudaMemcpyHostToDevice);

    scan_block_recursive(d_A, N);

    cudaMemcpy(&last_val, d_A+N-1, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_A);

    cout << "Last Val: " << last_val << endl;

    return 0;
}
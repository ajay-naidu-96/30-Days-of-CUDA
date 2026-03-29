#include <iostream>
#include <vector>
#define N 100000000
#define KERNEL_WIDTH 5
#define BLOCK_SIZE 1024

using namespace std;


__global__
void conv1D(const int *A, int *K, int *B) {

    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if (gid >= N) return;
    int total = 0;
    for (int i = -(KERNEL_WIDTH/2); i <= (KERNEL_WIDTH/2); i++) {
        if ((gid + i >= 0) && (gid + i < N)) {
            total += A[gid+i] * K[i+(KERNEL_WIDTH/2)];
        }
    }
    B[gid] = total;

}


int main() {

    int kernel[KERNEL_WIDTH];
    vector<int>A(N);
    vector<int>B(10);
    int *d_A, *d_K, *d_B;

    for (size_t i=0; i<KERNEL_WIDTH; i++) {
        kernel[i] = rand() % 100;
    }

    for (size_t i=0; i<N; i++) {
        A[i] = rand() % 999;
    }

    cudaMalloc(&d_A, N*sizeof(int));
    cudaMalloc(&d_B, N*sizeof(int));
    cudaMalloc(&d_K, KERNEL_WIDTH*sizeof(int));

    cudaMemcpy(d_A, A.data(), N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, kernel, KERNEL_WIDTH*sizeof(int), cudaMemcpyHostToDevice);

    size_t NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    conv1D<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_A, d_K, d_B);

    cudaDeviceSynchronize(); 

    cudaMemcpy(B.data(), d_B, sizeof(int)*10, cudaMemcpyDeviceToHost);

    cout << "Input: ";
    for (size_t i = 0; i < 10; i++) cout << A[i] << ", ";
    cout << endl;

    cout << "Kernel: ";
    for (size_t i = 0; i < 5; i++) cout << kernel[i] << ", ";
    cout << endl;

    cout << "Output: ";
    for (size_t i = 0; i < 10; i++) cout << B[i] << ", ";
    cout << endl;

    cudaFree(d_B);
    cudaFree(d_A);
    cudaFree(d_K);
    
    return 0;

}
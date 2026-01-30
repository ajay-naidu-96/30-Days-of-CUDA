#include <iostream>
#include <cuda_runtime.h>

#define ROWS 1024
#define COLS 1024

using namespace std;

__global__
void mat_transpose(int *A) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x >= ROWS) || (y >= COLS)) {
        return;
    }

    if (y > x) {
        return;
    }

    size_t p1 = y * COLS + x;
    size_t p2 = x * ROWS + y;

    int tmp1 = A[p1];
    int tmp2 = A[p2];
    A[p2] = tmp1;
    A[p1] = tmp2;

}


int main() {

    size_t mem_size = ROWS * COLS * sizeof(int);
    int *A = (int *)malloc(mem_size);
    int *d_A;

    for (size_t i=0; i<(ROWS*COLS); i++) {
        A[i] = i+1;
    }

    cout << "Pre Transpose: [";
    for (size_t i=0; i<5; i++) {
        cout << A[i] << ", ";
    }
    cout << "]" << endl;


    cudaMalloc(&d_A, mem_size);
    cudaMemcpy(d_A, A, mem_size, cudaMemcpyHostToDevice);
    
    int width = 32;
    int height = 32;

    dim3 block(width, height);
    dim3 grid((ROWS+width-1)/width, (COLS+height-1)/height);

    mat_transpose<<<grid, block>>> (d_A);  
    
    cudaMemcpy(A, d_A, mem_size, cudaMemcpyDeviceToHost);

    cout << "Post Transpose: [";
    for (size_t i=0; i<5; i++) {
        cout << A[i] << ", ";
    }
    cout << "]" << endl;

    cudaFree(d_A);
    free(A);

}


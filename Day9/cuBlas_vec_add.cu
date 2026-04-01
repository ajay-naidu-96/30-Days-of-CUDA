#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

using namespace std;

int main() {
    
    size_t n = 100;
    float *A = new float[n];
    float *B = new float[n];
    float *C = new float[n];

    for (size_t i=0; i < n; i++) {
        A[i] = rand()%100;
        B[i] = rand()%100;
        C[i] = B[i];
    }

    float *dA, *dB;
    float alpha = 1.0;

    cublasHandle_t h;
    cublasCreate(&h);

    cudaMalloc(&dA, sizeof(float)*n);
    cudaMalloc(&dB, sizeof(float)*n);

    cudaMemcpy(dA, A, sizeof(float)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(float)*n, cudaMemcpyHostToDevice);

    cublasSaxpy(h, n, &alpha, dA, 1, dB, 1);

    cudaMemcpy(B, dB, sizeof(float)*n, cudaMemcpyDeviceToHost);

    for (size_t i=0; i< 10; i++) {
        cout << "a: " << A[i] << ", C: " << C[i] << " = " << B[i] << endl;
    }

    delete[] A;
    delete[] B;
    delete[] C;

    cudaFree(dA);
    cudaFree(dB);

    cublasDestroy(h);

    return 0;

}
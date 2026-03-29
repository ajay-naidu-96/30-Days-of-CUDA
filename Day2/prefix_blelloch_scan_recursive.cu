#include <iostream>
#include <vector>
#define N 100000000
#define BLOCK_SIZE 1024

__global__
void scan_block(int *A, int n) {

}

void scan_recursive(int *A, int n) {
    if (n <= 1) return;

    if (n < BLOCK_SIZE) {
        scan_block_blelloch<<<1, BLOCK_SIZE>>>(A, *nullptr, n);
    }

    int num_block = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
}

int main() {
    vector<int> A(N, 1);
    scan_recursive(A.data(), N)
}
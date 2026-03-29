#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define BLOCK_SIZE 256
const int N = 100000000; // 100 Million elements

// ------------------------------------------------------------------
// 1. Block-Sequential Scan (Thread 0 does all work in a block)
// ------------------------------------------------------------------
__global__ void block_linear_scan(const int* A, int* P, int* blockSums, int n) {
    int bid = blockIdx.x;
    int start = bid * blockDim.x;
    if (start >= n) return;

    if (threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < blockDim.x && (start + i) < n; i++) {
            sum += A[start + i];
            P[start + i] = sum;
        }
        if (blockSums) blockSums[bid] = sum;
    }
}

// ------------------------------------------------------------------
// 2. Blelloch Scan (Work-efficient parallel scan per block)
// ------------------------------------------------------------------
__global__ void blelloch_block_scan(const int* A, int* P, int* blockSums, int n) {
    __shared__ int temp[BLOCK_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    int val = (gid < n) ? A[gid] : 0;
    temp[tid] = val;
    __syncthreads();

    // Upsweep (Reduction)
    for (int offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
        int i = (tid + 1) * offset * 2 - 1;
        if (i < BLOCK_SIZE) temp[i] += temp[i - offset];
        __syncthreads();
    }

    // Save block sum (total of all elements in block)
    int blockSum = temp[BLOCK_SIZE - 1];
    if (tid == 0 && blockSums) blockSums[blockIdx.x] = blockSum;

    if (tid == 0) temp[BLOCK_SIZE - 1] = 0;
    __syncthreads();

    // Downsweep
    for (int offset = BLOCK_SIZE >> 1; offset > 0; offset >>= 1) {
        int i = (tid + 1) * offset * 2 - 1;
        if (i < BLOCK_SIZE) {
            int t = temp[i - offset];
            temp[i - offset] = temp[i];
            temp[i] += t;
        }
        __syncthreads();
    }

    // Convert exclusive to inclusive scan by adding original value
    if (gid < n) P[gid] = temp[tid] + val;
}

// ------------------------------------------------------------------
// Helper: Add offsets back to blocks
// ------------------------------------------------------------------
__global__ void add_offsets(int* P, const int* offsets, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n && blockIdx.x > 0) {
        P[gid] += offsets[blockIdx.x - 1];
    }
}

// ------------------------------------------------------------------
// OLD: Serial scan (for comparison)
// ------------------------------------------------------------------
__global__ void scan_block_sums_serial(int* data, int numBlocks) {
    if (threadIdx.x == 0) {
        for (int i = 1; i < numBlocks; i++) {
            data[i] += data[i - 1];
        }
    }
}

// ------------------------------------------------------------------
// NEW: Recursive parallel scan for any array size
// ------------------------------------------------------------------
void scan_array_recursive_blelloch(int* d_data, int n) {
    if (n <= 1) return;
    
    if (n <= BLOCK_SIZE) {
        // Single block can handle it
        blelloch_block_scan<<<1, BLOCK_SIZE>>>(d_data, d_data, nullptr, n);
        return;
    }
    
    // Multiple blocks needed
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int *d_blockSums;
    cudaMalloc(&d_blockSums, numBlocks * sizeof(int));
    
    // Step 1: Scan each block independently
    blelloch_block_scan<<<numBlocks, BLOCK_SIZE>>>(d_data, d_data, d_blockSums, n);
    
    // Step 2: Recursively scan the block sums
    scan_array_recursive_blelloch(d_blockSums, numBlocks);
    
    // Step 3: Add block offsets to all elements
    add_offsets<<<numBlocks, BLOCK_SIZE>>>(d_data, d_blockSums, n);
    
    cudaFree(d_blockSums);
}

void scan_array_recursive_sequential(int* d_data, int n) {
    if (n <= 1) return;
    
    if (n <= BLOCK_SIZE) {
        // Single block can handle it
        block_linear_scan<<<1, BLOCK_SIZE>>>(d_data, d_data, nullptr, n);
        return;
    }
    
    // Multiple blocks needed
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int *d_blockSums;
    cudaMalloc(&d_blockSums, numBlocks * sizeof(int));
    
    // Step 1: Scan each block independently
    block_linear_scan<<<numBlocks, BLOCK_SIZE>>>(d_data, d_data, d_blockSums, n);
    
    // Step 2: Recursively scan the block sums
    scan_array_recursive_sequential(d_blockSums, numBlocks);
    
    // Step 3: Add block offsets to all elements
    add_offsets<<<numBlocks, BLOCK_SIZE>>>(d_data, d_blockSums, n);
    
    cudaFree(d_blockSums);
}

// ------------------------------------------------------------------
// MAIN
// ------------------------------------------------------------------
int main() {
    size_t size = N * sizeof(int);
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    std::vector<int> h_A(N, 1);
    int *d_A, *d_P;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_P, size);
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_ms = 0;

    std::cout << "Array size: " << N << " elements\n";
    std::cout << "Number of blocks: " << numBlocks << "\n\n";

    // ========== OLD IMPLEMENTATION (with serial bottleneck) ==========
    
    std::cout << "=== OLD IMPLEMENTATION (Serial Block-Sum Scan) ===\n";
    
    // --- TEST 1: BLOCK-SEQUENTIAL (OLD) ---
    int *d_blockSums;
    cudaMalloc(&d_blockSums, numBlocks * sizeof(int));
    cudaMemset(d_P, 0, size);
    cudaMemset(d_blockSums, 0, numBlocks * sizeof(int));
    
    cudaEventRecord(start);
    block_linear_scan<<<numBlocks, BLOCK_SIZE>>>(d_A, d_P, d_blockSums, N);
    scan_block_sums_serial<<<1, 1>>>(d_blockSums, numBlocks); // BOTTLENECK
    add_offsets<<<numBlocks, BLOCK_SIZE>>>(d_P, d_blockSums, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    int last_val;
    cudaMemcpy(&last_val, &d_P[N-1], sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&time_ms, start, stop);
    std::cout << "Block-sequential (OLD): " << time_ms << " ms | Last Element: " << last_val << "\n";

    // --- TEST 2: BLELLOCH (OLD) ---
    cudaMemset(d_P, 0, size);
    cudaMemset(d_blockSums, 0, numBlocks * sizeof(int));

    cudaEventRecord(start);
    blelloch_block_scan<<<numBlocks, BLOCK_SIZE>>>(d_A, d_P, d_blockSums, N);
    scan_block_sums_serial<<<1, 1>>>(d_blockSums, numBlocks); // BOTTLENECK
    add_offsets<<<numBlocks, BLOCK_SIZE>>>(d_P, d_blockSums, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(&last_val, &d_P[N-1], sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&time_ms, start, stop);
    std::cout << "Blelloch Parallel (OLD): " << time_ms << " ms | Last Element: " << last_val << "\n";

    cudaFree(d_blockSums);

    // ========== NEW IMPLEMENTATION (recursive parallel) ==========
    
    std::cout << "\n=== NEW IMPLEMENTATION (Recursive Parallel Scan) ===\n";
    
    // --- TEST 3: BLOCK-SEQUENTIAL (NEW) ---
    cudaMemcpy(d_P, d_A, size, cudaMemcpyDeviceToDevice);
    
    cudaEventRecord(start);
    scan_array_recursive_sequential(d_P, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaMemcpy(&last_val, &d_P[N-1], sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&time_ms, start, stop);
    std::cout << "Block-sequential (NEW): " << time_ms << " ms | Last Element: " << last_val << "\n";

    // --- TEST 4: BLELLOCH (NEW) ---
    cudaMemcpy(d_P, d_A, size, cudaMemcpyDeviceToDevice);

    cudaEventRecord(start);
    scan_array_recursive_blelloch(d_P, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(&last_val, &d_P[N-1], sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&time_ms, start, stop);
    std::cout << "Blelloch Parallel (NEW): " << time_ms << " ms | Last Element: " << last_val << "\n";

    // Cleanup
    cudaFree(d_A); 
    cudaFree(d_P);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}

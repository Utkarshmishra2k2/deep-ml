#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 256;

__global__ void kernel(const float* in, float* out, int N) {
    __shared__ float partial[BLOCK_SIZE];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    partial[tid] = (index < N) ? in[index] : 0.0f;
    __syncthreads();

    // Each step combines pairs of partial sums.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    // One sum per block.
    if (tid == 0) {
        out[blockIdx.x] = partial[0];
    }
}

void solve(const float* input, float* output, int N) {
    if (N <= 0) {
        *output = 0.0f;
        return;
    }

    float* d_input = nullptr;
    float* d_temp1 = nullptr;
    float* d_temp2 = nullptr;

    int max_blocks = (N - 1) / BLOCK_SIZE + 1;

    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_temp1, max_blocks * sizeof(float));
    cudaMalloc((void**)&d_temp2, max_blocks * sizeof(float));

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    const float* src = d_input;
    float* dst = d_temp1;
    int count = N;

    // Keep reducing block sums until only one remains.
    while (true) {
        int blocks = (count - 1) / BLOCK_SIZE + 1;
        kernel<<<blocks, BLOCK_SIZE>>>(src, dst, count);

        count = blocks;
        src = dst;
        dst = (dst == d_temp1) ? d_temp2 : d_temp1;

        if (count == 1) {
            break;
        }
    }

    cudaMemcpy(output, src, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_temp1);
    cudaFree(d_temp2);
}

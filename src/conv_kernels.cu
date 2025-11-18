
#include "conv_kernels.cuh"

// Simple SAXPY kernel: z[i] = a * x[i] + y[i]
__global__ void saxpy_kernel(const float* __restrict__ x,
                             const float* __restrict__ y,
                             float*       __restrict__ z,
                             float        a,
                             int          n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        z[idx] = a * x[idx] + y[idx];
    }
}

// Naive 3x3 box blur on a single-channel image
__global__ void blur3x3_naive(const float* __restrict__ in,
                              float*       __restrict__ out,
                              int          H,
                              int          W)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row

    if (x >= W || y >= H) {
        return;
    }

    // Skip 1-pixel border for simplicity
    if (x == 0 || x == W - 1 || y == 0 || y == H - 1) {
        out[y * W + x] = in[y * W + x];
        return;
    }

    float sum = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int yy = y + dy;
            int xx = x + dx;
            sum += in[yy * W + xx];
        }
    }
    out[y * W + x] = sum / 9.0f;
}

#include <cuda_runtime.h>
#include <cstdio>

extern "C" {

__global__ void saxpy_kernel(const float* __restrict__ x,
                             const float* __restrict__ y,
                             float* __restrict__ z,
                             float a,
                             int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = a * x[i] + y[i];
    }
}

// 2D 3x3 blur (naive, no shared memory yet) for practice
__global__ void blur3x3_naive(const float* __restrict__ in,
                              float* __restrict__ out,
                              int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float acc = 0.f;
    int count = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                acc += in[ny * W + nx];
                ++count;
            }
        }
    }
    out[y * W + x] = acc / (float)count;
}

} // extern "C"

#pragma once

#include <cuda_runtime.h>

// Simple SAXPY: z = a * x + y
__global__ void saxpy_kernel(const float* __restrict__ x,
                             const float* __restrict__ y,
                             float*       __restrict__ z,
                             float        a,
                             int          n);

// Naive 3x3 box blur on a single-channel H×W image
__global__ void blur3x3_naive(const float* __restrict__ in,
                              float*       __restrict__ out,
                              int          H,
                              int          W);

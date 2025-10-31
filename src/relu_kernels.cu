
#include "relu_kernels.cuh"
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    asm("trap;"); \
  } \
} while(0)

__global__ void relu_inplace(float* __restrict__ d, size_t n){
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<n){
    float x = d[i];
    d[i] = x > 0.f ? x : 0.f;
  }
}

void launch_relu_inplace(float* d, size_t n){
  const int TPB = 256;
  int blocks = (int)((n + TPB - 1) / TPB);
  relu_inplace<<<blocks, TPB>>>(d, n);
}

float gbps_relu(size_t n, float ms){
  // reads + writes of one float each
  double bytes = 2.0 * double(n) * sizeof(float);
  return float(bytes / (ms * 1e-3) / 1e9);
}

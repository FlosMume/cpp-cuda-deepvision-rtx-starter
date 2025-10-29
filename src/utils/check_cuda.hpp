
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                                  \
  cudaError_t err__ = (call);                                   \
  if (err__ != cudaSuccess) {                                   \
    fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
            cudaGetErrorString(err__), __FILE__, __LINE__);     \
    std::exit(EXIT_FAILURE);                                    \
  }                                                             \
} while (0)

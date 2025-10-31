
#pragma once
#include <cstddef>

void launch_relu_inplace(float* d, size_t n);
float gbps_relu(size_t n, float ms);

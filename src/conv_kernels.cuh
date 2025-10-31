
#pragma once
// NCHW naive conv forward + bias + ReLU-ready output
void launch_conv2d_naive_nchw(
  const float* x, const float* w, const float* b, float* y,
  int N, int C_in, int H, int W,
  int C_out, int K_h, int K_w,
  int stride_h, int stride_w,
  int pad_h, int pad_w);

// Rough FLOPs estimator for conv2d (forward)
float gflops_conv2d(int N,int C_in,int H,int W,int C_out,int K_h,int K_w,int stride_h,int stride_w,float ms);

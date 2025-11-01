
#pragma once
#ifdef HAVE_CUDNN
void cudnn_conv_relu_forward_nchw(
  const float* x, const float* w, const float* b, float* y,
  int N,int C_in,int H,int W,
  int C_out,int K_h,int K_w,
  int stride_h,int stride_w,
  int pad_h,int pad_w);
#else
inline void cudnn_conv_relu_forward_nchw(
  const float*, const float*, const float*, float*,
  int,int,int,int,int,int,int,int,int,int,int,int){}
#endif

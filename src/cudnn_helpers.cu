
#include "cudnn_helpers.cuh"
#ifdef HAVE_CUDNN
#include <cudnn.h>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#define CUDNN_CHECK(x) do { \
  cudnnStatus_t s = (x); \
  if(s != CUDNN_STATUS_SUCCESS){ \
    fprintf(stderr, "cuDNN error %s at %s:%d\n", cudnnGetErrorString(s), __FILE__, __LINE__); \
    asm("trap;"); \
  } \
} while(0)

void cudnn_conv_relu_forward_nchw(
  const float* x, const float* w, const float* b, float* y,
  int N,int C_in,int H,int W,
  int C_out,int K_h,int K_w,
  int stride_h,int stride_w,
  int pad_h,int pad_w)
{
  cudnnHandle_t h; CUDNN_CHECK(cudnnCreate(&h));

  cudnnTensorDescriptor_t xDesc, yDesc, bDesc;
  cudnnFilterDescriptor_t wDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnActivationDescriptor_t actDesc;

  CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&bDesc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&wDesc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));
  CUDNN_CHECK(cudnnCreateActivationDescriptor(&actDesc));

  // NCHW float
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_in, H, W));
  int H_out = (H + 2*pad_h - K_h)/stride_h + 1;
  int W_out = (W + 2*pad_w - K_w)/stride_w + 1;
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_out, H_out, W_out));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, C_out, C_in, K_h, K_w));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, stride_h, stride_w, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

  // Choose algo
  cudnnConvolutionFwdAlgo_t algo;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(h, xDesc, wDesc, convDesc, yDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

  // Workspace
  size_t ws_bytes=0;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(h, xDesc, wDesc, convDesc, yDesc, algo, &ws_bytes));
  void* ws=nullptr; if(ws_bytes) cudaMalloc(&ws, ws_bytes);

  // Bias
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(bDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C_out, 1, 1));

  // Activation (ReLU)
  CUDNN_CHECK(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));

  // conv
  const float alpha=1.f, beta=0.f;
  CUDNN_CHECK(cudnnConvolutionForward(h, &alpha, xDesc, x, wDesc, w, convDesc, algo, ws, ws_bytes, &beta, yDesc, y));

  // bias add
  CUDNN_CHECK(cudnnAddTensor(h, &alpha, bDesc, b, &alpha, yDesc, y));

  // relu in-place
  CUDNN_CHECK(cudnnActivationForward(h, actDesc, &alpha, yDesc, y, &beta, yDesc, y));

  if(ws) cudaFree(ws);
  cudnnDestroyActivationDescriptor(actDesc);
  cudnnDestroyConvolutionDescriptor(convDesc);
  cudnnDestroyFilterDescriptor(wDesc);
  cudnnDestroyTensorDescriptor(bDesc);
  cudnnDestroyTensorDescriptor(yDesc);
  cudnnDestroyTensorDescriptor(xDesc);
  cudnnDestroy(h);
}
#endif

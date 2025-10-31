
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cassert>
#include <cuda_runtime.h>

#include "relu_kernels.cuh"
#include "conv_kernels.cuh"
#include "cudnn_helpers.cuh"

#define CHECK_CUDA(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

static float elapsed_ms(cudaEvent_t a, cudaEvent_t b) {
  float ms=0.f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, a, b));
  return ms;
}

template<typename T>
T* dalloc(size_t n){
  T* p=nullptr;
  CHECK_CUDA(cudaMalloc(&p, n*sizeof(T)));
  return p;
}

float l2_err(const std::vector<float>& a, const std::vector<float>& b){
  assert(a.size()==b.size());
  double s=0.0, t=0.0;
  for(size_t i=0;i<a.size();++i){ double d = double(a[i])-double(b[i]); s += d*d; t += double(b[i])*double(b[i]); }
  return (float)std::sqrt(s / (t + 1e-20));
}

int main(){
  // Problem config (edit as needed)
  const int N=1;           // batch
  const int C_in=64;       // input channels
  const int H=128, W=128;  // input spatial
  const int C_out=64;      // output channels
  const int K=3;           // kernel size (square)
  const int stride=1;
  const int pad=1;         // same padding for K=3, stride=1

  const size_t in_elems  = (size_t)N*C_in*H*W;
  const size_t w_elems   = (size_t)C_out*C_in*K*K;
  const int H_out = (H + 2*pad - K)/stride + 1;
  const int W_out = (W + 2*pad - K)/stride + 1;
  const size_t out_elems = (size_t)N*C_out*H_out*W_out;

  printf("Config: N=%d C_in=%d HxW=%dx%d -> C_out=%d K=%d stride=%d pad=%d -> HxW_out=%dx%d\n",
         N, C_in, H, W, C_out, K, stride, pad, H_out, W_out);

  // Host buffers
  std::vector<float> h_x(in_elems), h_w(w_elems), h_b(C_out, 0.0f);
  std::vector<float> h_y_naive(out_elems), h_y_relu(out_elems), h_y_lib(out_elems);

  // Init random
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  for(auto& v : h_x) v = U(rng);
  for(auto& v : h_w) v = U(rng);
  for(auto& v : h_b) v = U(rng);

  // Device buffers
  float *d_x = dalloc<float>(in_elems);
  float *d_w = dalloc<float>(w_elems);
  float *d_b = dalloc<float>(C_out);
  float *d_y = dalloc<float>(out_elems);

  CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), in_elems*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), w_elems*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), C_out*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_y, 0, out_elems*sizeof(float)));

  cudaEvent_t t0,t1; CHECK_CUDA(cudaEventCreate(&t0)); CHECK_CUDA(cudaEventCreate(&t1));

  // --- Custom conv (naive) ---
  CHECK_CUDA(cudaEventRecord(t0));
  launch_conv2d_naive_nchw(d_x, d_w, d_b, d_y,
                           N, C_in, H, W,
                           C_out, K, K, stride, stride, pad, pad);
  CHECK_CUDA(cudaEventRecord(t1));
  CHECK_CUDA(cudaEventSynchronize(t1));
  float ms_naive = elapsed_ms(t0,t1);
  CHECK_CUDA(cudaMemcpy(h_y_naive.data(), d_y, out_elems*sizeof(float), cudaMemcpyDeviceToHost));
  printf("Custom conv (naive) time: %.3f ms  (%.2f GFLOP/s approx)\n",
         ms_naive, gflops_conv2d(N,C_in,H,W,C_out,K,K,stride,stride,ms_naive));

  // --- ReLU ---
  CHECK_CUDA(cudaEventRecord(t0));
  launch_relu_inplace(d_y, out_elems);
  CHECK_CUDA(cudaEventRecord(t1));
  CHECK_CUDA(cudaEventSynchronize(t1));
  float ms_relu = elapsed_ms(t0,t1);
  CHECK_CUDA(cudaMemcpy(h_y_relu.data(), d_y, out_elems*sizeof(float), cudaMemcpyDeviceToHost));
  printf("Custom ReLU time: %.3f ms (%.2f GB/s effective)\n",
         ms_relu, gbps_relu(out_elems, ms_relu));

  // --- cuDNN baseline (if available) ---
  bool have_cudnn=false;
#ifdef HAVE_CUDNN
  have_cudnn=true;
  CHECK_CUDA(cudaMemset(d_y, 0, out_elems*sizeof(float)));
  CHECK_CUDA(cudaEventRecord(t0));
  cudnn_conv_relu_forward_nchw(d_x, d_w, d_b, d_y,
                               N,C_in,H,W,C_out,K,K,stride,stride,pad,pad);
  CHECK_CUDA(cudaEventRecord(t1));
  CHECK_CUDA(cudaEventSynchronize(t1));
  float ms_cudnn = elapsed_ms(t0,t1);
  CHECK_CUDA(cudaMemcpy(h_y_lib.data(), d_y, out_elems*sizeof(float), cudaMemcpyDeviceToHost));
  printf("[cuDNN] conv+ReLU time: %.3f ms\n", ms_cudnn);
#endif

  // --- Correctness check (vs cuDNN if present; else vs self no-op) ---
  if(have_cudnn){
    float err = l2_err(h_y_relu, h_y_lib);
    printf("Relative L2 err vs cuDNN: %.3e\n", err);
  }else{
    printf("cuDNN not available; skipped baseline comparison.\n");
  }

  // Cleanup
  CHECK_CUDA(cudaFree(d_x)); CHECK_CUDA(cudaFree(d_w));
  CHECK_CUDA(cudaFree(d_b)); CHECK_CUDA(cudaFree(d_y));
  CHECK_CUDA(cudaEventDestroy(t0)); CHECK_CUDA(cudaEventDestroy(t1));

  return 0;
}

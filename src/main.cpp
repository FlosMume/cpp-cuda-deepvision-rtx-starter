
#include <cstdio>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "utils/check_cuda.hpp"

extern "C" {
void saxpy_kernel(const float* x, const float* y, float* z, float a, int n);
void blur3x3_naive(const float* in, float* out, int H, int W);
}

static void eventElapsedMS(cudaEvent_t beg, cudaEvent_t end, const char* tag) {
    float ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, beg, end));
    printf("[TIME] %s: %.3f ms\n", tag, ms);
}

int main(int argc, char** argv) {
    // Config
    int N = 1 << 24; // ~16.7M elements, ~64MB per float array
    int H = 2048, W = 2048; // ~4.2M pixels

    // Create streams
    const int N_STREAMS = 3;
    std::vector<cudaStream_t> streams(N_STREAMS);
    for (int s = 0; s < N_STREAMS; ++s) CHECK_CUDA(cudaStreamCreate(&streams[s]));

    // Events for timing
    cudaEvent_t e_start, e_after_h2d, e_after_k, e_after_d2h;
    CHECK_CUDA(cudaEventCreate(&e_start));
    CHECK_CUDA(cudaEventCreate(&e_after_h2d));
    CHECK_CUDA(cudaEventCreate(&e_after_k));
    CHECK_CUDA(cudaEventCreate(&e_after_d2h));

    // Allocate pinned host memory for overlap
    float *hx, *hy, *hz;
    size_t bytes = N * sizeof(float);
    CHECK_CUDA(cudaMallocHost(&hx, bytes));
    CHECK_CUDA(cudaMallocHost(&hy, bytes));
    CHECK_CUDA(cudaMallocHost(&hz, bytes));
    for (int i = 0; i < N; ++i) { hx[i] = 1.f; hy[i] = 2.f; }

    // Device buffers
    float *dx, *dy, *dz;
    CHECK_CUDA(cudaMalloc(&dx, bytes));
    CHECK_CUDA(cudaMalloc(&dy, bytes));
    CHECK_CUDA(cudaMalloc(&dz, bytes));

    // Begin timing
    CHECK_CUDA(cudaEventRecord(e_start, streams[0]));

    // Async H2D copies on streams 0 and 1
    CHECK_CUDA(cudaMemcpyAsync(dx, hx, bytes, cudaMemcpyHostToDevice, streams[0]));
    CHECK_CUDA(cudaMemcpyAsync(dy, hy, bytes, cudaMemcpyHostToDevice, streams[1]));

    // Record after H2D (use stream 0 for timing)
    CHECK_CUDA(cudaEventRecord(e_after_h2d, streams[0]));

    // Launch saxpy on stream 2 to overlap with copies (some overlap may still occur depending on HW)
    const float a = 3.f;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    saxpy_kernel<<<blocks, threads, 0, streams[2]>>>(dx, dy, dz, a, N);
    CHECK_CUDA(cudaGetLastError());

    // Record after kernel
    CHECK_CUDA(cudaEventRecord(e_after_k, streams[2]));

    // Async D2H
    CHECK_CUDA(cudaMemcpyAsync(hz, dz, bytes, cudaMemcpyDeviceToHost, streams[0]));

    // Record after D2H
    CHECK_CUDA(cudaEventRecord(e_after_d2h, streams[0]));

    // Synchronize all streams
    for (auto& s : streams) CHECK_CUDA(cudaStreamSynchronize(s));

    // Validate a couple values
    printf("hz[0]=%.1f  hz[N-1]=%.1f  (expect 5.0)\n", hz[0], hz[N-1]);

    // Timings
    eventElapsedMS(e_start, e_after_h2d, "H2D async copies");
    eventElapsedMS(e_after_h2d, e_after_k, "Kernel saxpy");
    eventElapsedMS(e_after_k, e_after_d2h, "D2H async copy");
    eventElapsedMS(e_start, e_after_d2h, "End-to-end");

    // Simple 2D blur to practice 2D grids
    size_t img_bytes = H * W * sizeof(float);
    float *h_img_in, *h_img_out, *d_img_in, *d_img_out;
    CHECK_CUDA(cudaMallocHost(&h_img_in, img_bytes));
    CHECK_CUDA(cudaMallocHost(&h_img_out, img_bytes));
    for (int i = 0; i < H*W; ++i) h_img_in[i] = (i % 255) / 255.0f;

    CHECK_CUDA(cudaMalloc(&d_img_in, img_bytes));
    CHECK_CUDA(cudaMalloc(&d_img_out, img_bytes));
    CHECK_CUDA(cudaMemcpy(d_img_in, h_img_in, img_bytes, cudaMemcpyHostToDevice));

    dim3 block2d(16, 16);
    dim3 grid2d((W + block2d.x - 1)/block2d.x, (H + block2d.y - 1)/block2d.y);
    blur3x3_naive<<<grid2d, block2d>>>(d_img_in, d_img_out, H, W);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(h_img_out, d_img_out, img_bytes, cudaMemcpyDeviceToHost));

    printf("Blur sample: in[0]=%.3f out[0]=%.3f\n", h_img_in[0], h_img_out[0]);

    // Cleanup
    CHECK_CUDA(cudaFree(d_img_in));
    CHECK_CUDA(cudaFree(d_img_out));
    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));
    CHECK_CUDA(cudaFree(dz));
    CHECK_CUDA(cudaFreeHost(hx));
    CHECK_CUDA(cudaFreeHost(hy));
    CHECK_CUDA(cudaFreeHost(hz));
    CHECK_CUDA(cudaFreeHost(h_img_in));
    CHECK_CUDA(cudaFreeHost(h_img_out));
    for (auto& s : streams) cudaStreamDestroy(s);
    cudaEventDestroy(e_start);
    cudaEventDestroy(e_after_h2d);
    cudaEventDestroy(e_after_k);
    cudaEventDestroy(e_after_d2h);

    printf("Done.\n");
    return 0;
}

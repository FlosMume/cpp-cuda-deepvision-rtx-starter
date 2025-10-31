
# DeepVision-RTX (Starter)

A CUDA C++ practice project designed for the RTX 4070 SUPER (Ada 8.9), demonstrating how to overlap data transfers with computation using streams and pinned memory, apply basic kernel optimizations with 1D and 2D grid configurations, and perform precise event timing for profiling in Nsight Systems and Nsight Compute.

## What’s here?
- **Pinned host memory** + `cudaMemcpyAsync` to demonstrate overlap
- **Multiple streams** for concurrent copy/compute
- **Timed sections** with `cudaEventRecord` / `cudaEventElapsedTime`
- **Kernels**: `saxpy` (1D), `blur3x3_naive` (2D)

## Build
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/deepvision_rtx
```

## Profile (examples)
```bash
# Nsight Systems GUI (on host with CUDA Toolkit installed)
nsys profile -o nsys_report ./build/deepvision_rtx

# Nsight Compute single-kernel collection
ncu --set full --target-processes all ./build/deepvision_rtx
```

## Next steps
- Convert blur kernel to **shared-memory tiled** version
- Add **half-precision** path to prep for Tensor Cores
- Compare end-to-end with **cuDNN** and optionally TensorRT
```


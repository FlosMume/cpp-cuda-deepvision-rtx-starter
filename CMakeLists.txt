
cmake_minimum_required(VERSION 3.24)
project(deepvision_rtx LANGUAGES CXX CUDA)

# CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target GPU architectures - RTX 4070 SUPER (Ada) is 8.9
set(CMAKE_CUDA_ARCHITECTURES 89)

# Warnings
if (MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wno-unknown-pragmas)
endif()

# Sources
set(SRC
  src/main.cpp
  src/conv_kernels.cu
)

add_executable(deepvision_rtx ${SRC})

# Enable separable compilation
set_target_properties(deepvision_rtx PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Optimize by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

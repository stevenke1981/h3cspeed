# Optional convenience toolchain. Most WSL2 builds can use the default toolchain.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_C_COMPILER gcc CACHE STRING "")
set(CMAKE_CXX_COMPILER g++ CACHE STRING "")
set(CMAKE_CUDA_COMPILER nvcc CACHE STRING "")

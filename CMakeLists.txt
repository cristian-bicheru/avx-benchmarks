cmake_minimum_required(VERSION 3.15)
project(simd_benchmarks)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-mavx -falign-loops=32 -falign-functions=32 -march=native -O3 -mno-vzeroupper -funroll-all-loops -fprefetch-loop-arrays")

add_executable(main main.cpp)
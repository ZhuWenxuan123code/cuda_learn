cmake_minimum_required(VERSION 3.10)
project(MyCudaLearn LANGUAGES CXX CUDA) # 关键：启用CUDA语言支持

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_executable(00hello_cuda learn/00helloworld.cu)
add_executable(01detect_device learn/01detect_device.cu)
add_executable(03GPUMemorySpaces learn/03GPUMemorySpaces.cu)
add_executable(04Matrix_Transpose_Example learn/04Matrix_Transpose_Example.cu)
add_executable(05check_pageable_memory_access learn/05check_pageable_memory_access.cu)



add_executable(01-vector-add 01-vector-add.cu)
# 如果链接其他库，用 target_link_libraries
# target_link_libraries(${PROJECT_NAME} your_library)

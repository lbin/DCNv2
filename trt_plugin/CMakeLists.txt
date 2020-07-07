cmake_minimum_required(VERSION 3.10)

project(DCNv2_TRT_Plugin CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build type." FORCE)
  message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}'")
endif()

add_library(dcnv2_trt_plugin MODULE DeformConv.cpp DeformConv.cu)
target_compile_options(dcnv2_trt_plugin PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall>)
target_include_directories(dcnv2_trt_plugin PRIVATE /usr/local/cuda/include)
target_link_libraries(dcnv2_trt_plugin -lnvinfer -lnvinfer_plugin)

set_target_properties(dcnv2_trt_plugin PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

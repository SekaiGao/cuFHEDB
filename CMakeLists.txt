cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(FHEDB VERSION 1.0 LANGUAGES CXX CUDA)

message(STATUS "Initial CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUDA_NVCC_FLAGS "${CMAKE_CUDA_FLAGS} -march=native -std=c++17 -O3 -gencode=arch=compute_89,code=sm_89 -dlcm=ca -Xcompiler -fopenmp -fPIC --ftz=true --use_fast_math=true")

message(STATUS "Updated CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

# make details
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -finput-charset=UTF-8")
set(CMAKE_CXX_FLAGS "-march=native -std=c++17 -O3 -g -funroll-loops -Wno-sign-compare -Wno-unused-result -Wno-type-limits -Wno-sign-conversion -Wno-sign-compare")

find_package(CUDA)
message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
include_directories(${CUDA_INCLUDE_DIRS})


#--------------------------Source tree-----------------------------
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")


#------------------------------Install-----------------------------
set(FHEDB_INCLUDES_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}/FHEDB)


# -------------------------Add thirdparty--------------------------
add_subdirectory(thirdparty/TFHEpp)
add_subdirectory(thirdparty/SEAL)

# HE3DB
add_subdirectory(thirdparty/HEDB)
add_library(HEDB STATIC ${HEDB_SOURCE_FILES})
target_include_directories(HEDB PUBLIC src)
target_include_directories(HEDB PUBLIC thirdparty)
target_include_directories(HEDB PUBLIC thirdparty/SEAL/native/src)
target_link_libraries(HEDB tfhe++ seal)

# ARCEDB
add_subdirectory(thirdparty/ARCEDB)
add_library(ARCEDB STATIC ${ARCEDB_SOURCE_FILES})
target_include_directories(ARCEDB PUBLIC src)
target_include_directories(ARCEDB PUBLIC thirdparty)
target_include_directories(ARCEDB PUBLIC thirdparty/SEAL/native/src)
target_link_libraries(ARCEDB tfhe++ seal)

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# cuFHEDB
add_subdirectory(src)

add_subdirectory(test)

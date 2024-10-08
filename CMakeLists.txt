cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(HEDB VERSION 1.0 LANGUAGES CXX CUDA)

message(STATUS "Initial CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

set(CUDA_NVCC_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -O3 -gencode=arch=compute_86,code=sm_86 --ftz=true --use_fast_math=true --extra-device-vectorization --no-exception --allow-expensive-optimizations --ptxas-options=-O3 -Xptxas -dlcm=ca -Xcompiler -fopenmp")


message(STATUS "Updated CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

# make details
# set (CMAKE_VERBOSE_MAKEFILE on) 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native -O3 -g -funroll-loops -Wno-sign-compare -Wno-unused-result -Wno-shadow -Wno-type-limits -Wno-sign-conversion")


find_package(CUDA)
message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
include_directories(${CUDA_INCLUDE_DIRS})


#--------------------------Source tree-----------------------------
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")


#------------------------------Install-----------------------------
set(HEDB_INCLUDES_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}/HEDB)

# ------Add source files to library and header files to install----
set(HEDB_SOURCE_FILES "")
add_subdirectory(src/cuHEDB)
add_subdirectory(src/HEDB/comparison)
add_subdirectory(src/HEDB/conversion)
add_subdirectory(src/HEDB/utils)
# -------------------------Add thirdparty--------------------------
add_subdirectory(thirdparty/TFHEpp)
add_subdirectory(thirdparty/SEAL)
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#-------------------Build only a static library--------------------
add_library(HEDB STATIC ${HEDB_SOURCE_FILES}) 
target_include_directories(HEDB PUBLIC src)
target_include_directories(HEDB PUBLIC thirdparty/SEAL/natice/src)
target_link_libraries(HEDB tfhe++ seal)

# ARCEDB
set(ARCEDB_SOURCE_FILES "")
add_subdirectory(src/ARCEDB/utils)
add_subdirectory(src/ARCEDB/conversion)
add_subdirectory(src/ARCEDB/comparison)
# add_subdirectory(src/cuHEDB)

# Add ARCEDB library
add_library(ARCEDB STATIC ${ARCEDB_SOURCE_FILES})
target_include_directories(ARCEDB PUBLIC src)
target_include_directories(ARCEDB PUBLIC thirdparty/SEAL/natice/src)
target_link_libraries(ARCEDB tfhe++ seal)

add_subdirectory(test)

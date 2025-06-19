add_executable(TPCH_Q1 Database/TPC-H/TPCH_Q1.cu)
target_include_directories(TPCH_Q1 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(TPCH_Q1 ARCEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(TPCH_Q6 Database/TPC-H/TPCH_Q6.cu)
target_include_directories(TPCH_Q6 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(TPCH_Q6 ARCEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(TPCH_Q12 Database/TPC-H/TPCH_Q12.cu)
target_include_directories(TPCH_Q12 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(TPCH_Q12 ARCEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(TPCH_Q14 Database/TPC-H/TPCH_Q14.cu)
target_include_directories(TPCH_Q14 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(TPCH_Q14 ARCEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)


add_executable(HC_Q1 Database/Healthcare/HC_Q1.cu)
target_include_directories(HC_Q1 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(HC_Q1 ARCEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(HC_Q2 Database/Healthcare/HC_Q2.cu)
target_include_directories(HC_Q2 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(HC_Q2 ARCEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(HC_Q3 Database/Healthcare/HC_Q3.cu)
target_include_directories(HC_Q3 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(HC_Q3 ARCEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(HC_Q4 Database/Healthcare/HC_Q4.cu)
target_include_directories(HC_Q4 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(HC_Q4 ARCEDB HEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)


add_executable(comparison_test Database/comparison.cu)
target_include_directories(comparison_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(comparison_test ARCEDB HEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(batchfft_test FHE/batchfft.cu)
target_include_directories(batchfft_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(batchfft_test HEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(fft_test FHE/fft.cu)
target_include_directories(fft_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(fft_test HEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(externalproduct_test FHE/externalproduct.cu)
target_include_directories(externalproduct_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(externalproduct_test HEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(bootstrap_test FHE/bootstrapping.cu)
target_include_directories(bootstrap_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(bootstrap_test HEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)


add_executable(Arc_TPCH_Q1 Database/TPC-H/Baseline/Arc_TPCH_Q1.cpp)
target_include_directories(Arc_TPCH_Q1 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(Arc_TPCH_Q1 ARCEDB ${CUDA_LIBRARIES})

add_executable(Arc_TPCH_Q6 Database/TPC-H/Baseline/Arc_TPCH_Q6.cpp)
target_include_directories(Arc_TPCH_Q6 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(Arc_TPCH_Q6 ARCEDB ${CUDA_LIBRARIES})

add_executable(Arc_TPCH_Q12 Database/TPC-H/Baseline/Arc_TPCH_Q12.cpp)
target_include_directories(Arc_TPCH_Q12 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(Arc_TPCH_Q12 ARCEDB ${CUDA_LIBRARIES})

add_executable(Arc_TPCH_Q14 Database/TPC-H/Baseline/Arc_TPCH_Q14.cpp)
target_include_directories(Arc_TPCH_Q14 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(Arc_TPCH_Q14 ARCEDB ${CUDA_LIBRARIES})



add_executable(HE3_TPCH_Q1 Database/TPC-H/Baseline/HE3_TPCH_Q1.cpp)
target_include_directories(HE3_TPCH_Q1 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(HE3_TPCH_Q1 HEDB ${CUDA_LIBRARIES})

add_executable(HE3_TPCH_Q6 Database/TPC-H/Baseline/HE3_TPCH_Q6.cpp)
target_include_directories(HE3_TPCH_Q6 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(HE3_TPCH_Q6 HEDB ${CUDA_LIBRARIES})

add_executable(HE3_TPCH_Q12 Database/TPC-H/Baseline/HE3_TPCH_Q12.cpp)
target_include_directories(HE3_TPCH_Q12 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(HE3_TPCH_Q12 HEDB ${CUDA_LIBRARIES})

add_executable(HE3_TPCH_Q14 Database/TPC-H/Baseline/HE3_TPCH_Q14.cpp)
target_include_directories(HE3_TPCH_Q14 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(HE3_TPCH_Q14 HEDB ${CUDA_LIBRARIES})


add_executable(Arc_HC_Q1 Database/Healthcare/Baseline/Arc_HC_Q1.cpp)
target_include_directories(Arc_HC_Q1 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(Arc_HC_Q1 ARCEDB ${CUDA_LIBRARIES})

add_executable(Arc_HC_Q2 Database/Healthcare/Baseline/Arc_HC_Q2.cpp)
target_include_directories(Arc_HC_Q2 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(Arc_HC_Q2 ARCEDB ${CUDA_LIBRARIES})

add_executable(Arc_HC_Q3 Database/Healthcare/Baseline/Arc_HC_Q3.cpp)
target_include_directories(Arc_HC_Q3 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(Arc_HC_Q3 ARCEDB ${CUDA_LIBRARIES})

add_executable(Arc_HC_Q4 Database/Healthcare/Baseline/Arc_HC_Q4.cpp)
target_include_directories(Arc_HC_Q4 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(Arc_HC_Q4 ARCEDB HEDB ${CUDA_LIBRARIES})
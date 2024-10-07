add_executable(HE3TPCH_Q6 HE3_Query/TPCH_Q6.cu)
target_include_directories(HE3TPCH_Q6 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(HE3TPCH_Q6 HEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(HE3TPCH_Q4 HE3_Query/TPCH_Q4.cu)
target_include_directories(HE3TPCH_Q4 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(HE3TPCH_Q4 HEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(HE3TPCH_Q17 HE3_Query/TPCH_Q17.cu)
target_include_directories(HE3TPCH_Q17 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(HE3TPCH_Q17 HEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(HE3TPCH_Q12 HE3_Query/TPCH_Q12.cu)
target_include_directories(HE3TPCH_Q12 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(HE3TPCH_Q12 HEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(HE3TPCH_Q14 HE3_Query/TPCH_Q14.cu)
target_include_directories(HE3TPCH_Q14 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(HE3TPCH_Q14 HEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(HE3TPCH_Q1 HE3_Query/TPCH_Q1.cu)
target_include_directories(HE3TPCH_Q1 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(HE3TPCH_Q1 HEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(genkey GenKey.cu)
target_include_directories(genkey PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(genkey HEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(test10 HE3_Query/test1.cu)
target_include_directories(test10 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(test10 HEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(relational_queries_test Arc_Query/relational_queries_test.cpp)
target_include_directories(relational_queries_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(relational_queries_test ARCEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(ArcTPCH_Q1 Arc_Query/TPCH_Q1.cu)
target_include_directories(ArcTPCH_Q1 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(ArcTPCH_Q1 ARCEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(ArcTPCH_Q6 Arc_Query/TPCH_Q6.cu)
target_include_directories(ArcTPCH_Q6 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(ArcTPCH_Q6 ARCEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(ArcTPCH_Q12 Arc_Query/TPCH_Q12.cu)
target_include_directories(ArcTPCH_Q12 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(ArcTPCH_Q12 ARCEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(ArcTPCH_Q14 Arc_Query/TPCH_Q14.cu)
target_include_directories(ArcTPCH_Q14 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(ArcTPCH_Q14 ARCEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(test Arc_Query/test.cu)
target_include_directories(test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(test ARCEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(test1 Arc_Query/test1.cu)
target_include_directories(test1 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(test1 ARCEDB ${CUDA_LIBRARIES} cufft OpenMP::OpenMP_CXX)

add_executable(cufft_test cuFFT.cu)
target_include_directories(cufft_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cufft_test HEDB ${CUDA_LIBRARIES} cufft OpenMP::OpenMP_CXX)

add_executable(externalproduct_test ExternalProduct.cu)
target_include_directories(externalproduct_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(externalproduct_test HEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(bootstrap_test Bootstrapping.cu)
target_include_directories(bootstrap_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(bootstrap_test HEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(homcomp_test HE3_Query/HomComp.cu)
target_include_directories(homcomp_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(homcomp_test HEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(comparable_test Arc_Query/comparable.cu)
target_include_directories(comparable_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(comparable_test ARCEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(fusion_test kernelFusion.cu)
target_include_directories(fusion_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(fusion_test HEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)
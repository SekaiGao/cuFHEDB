add_executable(TPCH_Q6 Query/TPCH_Q6.cu)
target_include_directories(TPCH_Q6 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(TPCH_Q6 HEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(TPCH_Q4 Query/TPCH_Q4.cu)
target_include_directories(TPCH_Q4 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(TPCH_Q4 HEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(TPCH_Q17 Query/TPCH_Q17.cu)
target_include_directories(TPCH_Q17 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(TPCH_Q17 HEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(TPCH_Q12 Query/TPCH_Q12.cu)
target_include_directories(TPCH_Q12 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(TPCH_Q12 HEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(TPCH_Q14 Query/TPCH_Q14.cu)
target_include_directories(TPCH_Q14 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(TPCH_Q14 HEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(TPCH_Q1 Query/TPCH_Q1.cu)
target_include_directories(TPCH_Q1 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(TPCH_Q1 HEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

add_executable(genkey GenKey.cu)
target_include_directories(genkey PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(genkey HEDB ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX)

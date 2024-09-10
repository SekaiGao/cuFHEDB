# Source files in this directory
set(cuHEDB_SOURCE_FILES ${cuHEDB_SOURCE_FILES}
)

# Add header files for installation
install(
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/fft_gpu/fft_gpu.cuh
        ${CMAKE_CURRENT_LIST_DIR}/fft_gpu/ExternalProduct_gpu.cuh
        ${CMAKE_CURRENT_LIST_DIR}/fft_gpu/BlindRotate_gpu.cuh
        ${CMAKE_CURRENT_LIST_DIR}/fft_gpu/cufft_gpu.cuh
        ${CMAKE_CURRENT_LIST_DIR}/gatebootstrap_gpu.cuh
        ${CMAKE_CURRENT_LIST_DIR}/msb_gpu.cuh
        ${CMAKE_CURRENT_LIST_DIR}/HomCompare_gpu.cuh
    DESTINATION
    ${HEDB_INCLUDES_INSTALL_DIR}/cuHEDB
)

set(HEDB_SOURCE_FILES ${HEDB_SOURCE_FILES} PARENT_SCOPE)
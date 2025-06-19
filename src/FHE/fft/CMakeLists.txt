# Source files in this directory
set(cuFHEDB_SOURCE_FILES ${cuFHEDB_SOURCE_FILES}
)

# Add header files for installation
install(
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/fft/fft_gpu.cuh
		${CMAKE_CURRENT_LIST_DIR}/fft/ifft_gpu.cuh
    DESTINATION
    ${cuHEDB_INCLUDES_INSTALL_DIR}/FHE/fft
)

set(cuFHEDB_SOURCE_FILES ${cuFHEDB_SOURCE_FILES} PARENT_SCOPE)



# Source files in this directory
set(cuFHEDB_SOURCE_FILES ${cuFHEDB_SOURCE_FILES}
)

# Add header files for installation
install(
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/externalproduct/ExternalProduct_gpu.cuh
        ${CMAKE_CURRENT_LIST_DIR}/externalproduct/polymul.cuh
    DESTINATION
    ${cuHEDB_INCLUDES_INSTALL_DIR}/FHE/externalproduct
)

set(cuFHEDB_SOURCE_FILES ${cuFHEDB_SOURCE_FILES} PARENT_SCOPE)



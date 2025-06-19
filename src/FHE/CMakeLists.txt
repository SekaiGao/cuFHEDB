# Initialize cuFHEDB_SOURCE_FILES to an empty set
set(cuFHEDB_SOURCE_FILES "")

# Add subdirectories for different components in core FHE algorithm
add_subdirectory(fft)
add_subdirectory(externalproduct)
add_subdirectory(blindrotate)
add_subdirectory(bootstrapping)

install(
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/FHE/coreFHE.cuh
        ${CMAKE_CURRENT_LIST_DIR}/FHE/utils.cuh
    DESTINATION
    ${cuHEDB_INCLUDES_INSTALL_DIR}/FHE
)

# Optionally propagate cuFHEDB_SOURCE_FILES to the parent scope if needed
set(cuFHEDB_SOURCE_FILES ${cuFHEDB_SOURCE_FILES} PARENT_SCOPE)

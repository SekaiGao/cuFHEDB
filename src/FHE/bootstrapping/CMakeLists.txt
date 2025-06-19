# Source files in this directory
set(cuFHEDB_SOURCE_FILES ${cuFHEDB_SOURCE_FILES}
)

# Add header files for installation
install(
    FILES
    ${CMAKE_CURRENT_LIST_DIR}/bootstrapping/GateBootstrapping.cuh
    ${CMAKE_CURRENT_LIST_DIR}/bootstrapping/KeyBundle.cuh
    DESTINATION
    ${cuHEDB_INCLUDES_INSTALL_DIR}/coreFHE/bootstrapping
)

set(cuFHEDB_SOURCE_FILES ${cuFHEDB_SOURCE_FILES} PARENT_SCOPE)



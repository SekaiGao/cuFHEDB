set(cuFHEDB_SOURCE_FILES ${cuFHEDB_SOURCE_FILES})

# Add subdirectories for different FHE-based Database
add_subdirectory(cuFHEDB)
add_subdirectory(cuHEDB)

install(
    FILES
        ${CMAKE_CURRENT_LIST_DIR}/DatabaseParser.h
    DESTINATION
    ${cuHEDB_INCLUDES_INSTALL_DIR}/Database
)

set(cuFHEDB_SOURCE_FILES ${cuFHEDB_SOURCE_FILES} PARENT_SCOPE)

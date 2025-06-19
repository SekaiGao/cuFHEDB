# Initialize cuFHEDB_SOURCE_FILES to an empty set
set(cuFHEDB_SOURCE_FILES "")

add_subdirectory(FHE)
add_subdirectory(Database)

set(cuFHEDB_SOURCE_FILES ${cuFHEDB_SOURCE_FILES} PARENT_SCOPE)

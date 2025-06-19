set(HEDB_SOURCE_FILES "")

add_subdirectory(comparison)
add_subdirectory(conversion)
add_subdirectory(utils)

set(HEDB_SOURCE_FILES ${HEDB_SOURCE_FILES} PARENT_SCOPE)

# add_library(HEDB STATIC ${HEDB_SOURCE_FILES})
# target_include_directories(HEDB PUBLIC src)
# target_include_directories(HEDB PUBLIC thirdparty/SEAL/native/src)
# target_link_libraries(HEDB tfhe++ seal)
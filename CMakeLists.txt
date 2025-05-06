cmake_minimum_required(VERSION 3.10)
project(SPHiros LANGUAGES CXX)

enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable coverage flags for GCC/Clang
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(--coverage)
    add_link_options(--coverage)
endif()

# Find required packages
find_package(MPI REQUIRED)
find_package(Kokkos REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(CLI11 REQUIRED)
find_package(GTest REQUIRED)
find_package(Doxygen REQUIRED)

get_target_property(LIBRARY_TYPE yaml-cpp::yaml-cpp TYPE)

# Copy the examples folder to the build directory
file(COPY ${CMAKE_SOURCE_DIR}/examples DESTINATION ${CMAKE_BINARY_DIR})

# Source files
add_subdirectory(src)
# Test files
add_subdirectory(tests)

# Documentation# Configure Doxygen
set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

# Add a target to generate documentation
add_custom_target(doc
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM
)
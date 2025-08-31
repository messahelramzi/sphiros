# Create an interface library for the header-only files
add_library(IO INTERFACE)

# Specify include directories for this library
target_include_directories(IO INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Link required libraries to this library (if needed)
target_link_libraries(IO INTERFACE Kokkos::kokkos pybind11::embed Python3::Python)
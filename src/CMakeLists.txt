add_subdirectory(eos)
add_subdirectory(io)

# Add the executable
add_executable(SPHiros sphiros.cpp)

message(STATUS "Checking yaml-cpp library type...")
if(LIBRARY_TYPE STREQUAL "SHARED_LIBRARY")
  if(NOT YAML_CPP_SHARED_LIBS_BUILT)
    message(FATAL_ERROR "Library type (${LIBRARY_TYPE}) contradicts config: ${YAML_CPP_SHARED_LIBS_BUILT}")
  endif()
else()
  if(YAML_CPP_SHARED_LIBS_BUILT)
    message(FATAL_ERROR "Library type (${LIBRARY_TYPE}) contradicts config: ${YAML_CPP_SHARED_LIBS_BUILT}")
  endif()
endif()
message(STATUS "yaml-cpp library type check passed.")

# Link the required libraries
target_link_libraries(SPHiros PUBLIC EOS IO MPI::MPI_CXX Kokkos::kokkos yaml-cpp::yaml-cpp CLI11::CLI11 pybind11::embed Python3::Python)
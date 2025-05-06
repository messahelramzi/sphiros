function(add_eos_test target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE GTest::GTest GTest::Main Kokkos::kokkos EOS)
    gtest_discover_tests(${target_name})
endfunction()

add_eos_test(eos_linear_gas eos_linear_gas.cpp)
add_eos_test(eos_stiffened_gas eos_stiffened_gas.cpp)

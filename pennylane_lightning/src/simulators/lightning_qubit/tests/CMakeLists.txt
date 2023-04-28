cmake_minimum_required(VERSION 3.20)

project(pl_qubit_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${CMAKE_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define compile options and library
################################################################################

add_library(pl_qubit_tests INTERFACE)
target_link_libraries(pl_qubit_tests INTERFACE  Catch2::Catch2
                                                lightning_base
                                                lightning_utils
                                                lightning_qubit
                                                )

ProcessTestOptions(pl_qubit_tests)

target_sources(pl_qubit_tests INTERFACE runner_pl_qubit.cpp)

################################################################################
# Define targets
################################################################################

# add_executable(compile_time_tests compile_time_tests.cpp)
# target_link_libraries(compile_time_tests lightning_compile_options lightning_gates lightning_utils lightning_simulator)

set(TEST_SOURCES    Test_StateVectorLQubit.cpp
                    )

add_executable(pl_qubit_test_runner ${TEST_SOURCES})
target_link_libraries(pl_qubit_test_runner PRIVATE   pl_qubit_tests
                                                     lightning_compile_options
                                                     lightning_external_libs)
catch_discover_tests(pl_qubit_test_runner)

# We build compile time tests before the runtime tests as build error messages
# are horrible if compile time constants are not well defined.
# add_dependencies(pl_qubit_test_runner compile_time_tests)

install(TARGETS pl_qubit_test_runner DESTINATION bin)

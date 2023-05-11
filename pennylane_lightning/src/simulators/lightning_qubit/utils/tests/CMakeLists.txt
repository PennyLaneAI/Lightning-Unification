cmake_minimum_required(VERSION 3.20)

project(pl_qubit_utils_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${CMAKE_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(pl_qubit_utils_tests INTERFACE)
target_link_libraries(pl_qubit_utils_tests INTERFACE    Catch2::Catch2
                                                        pl_qubit_utils
                                                        )

ProcessTestOptions(pl_qubit_utils_tests)

target_sources(pl_qubit_utils_tests INTERFACE runner_pl_qubit_utils.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES    Test_RuntimeInfo.cpp
                    Test_LinearAlgebra.cpp
                    )

add_executable(pl_qubit_utils_test_runner ${TEST_SOURCES})
target_link_libraries(pl_qubit_utils_test_runner PRIVATE pl_qubit_utils_tests)
catch_discover_tests(pl_qubit_utils_test_runner)

install(TARGETS pl_qubit_utils_test_runner DESTINATION bin)

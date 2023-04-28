cmake_minimum_required(VERSION 3.20)

project(algorithms_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${CMAKE_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define compile options and library
################################################################################

add_library(algorithms_tests INTERFACE)
target_link_libraries(algorithms_tests INTERFACE    Catch2::Catch2
                                                    lightning_utils
                                                    lightning_algorithms
                                                    )

ProcessTestOptions(algorithms_tests)

target_sources(algorithms_tests INTERFACE runner_algorithms.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES    Test_AdjointJacobian.cpp
                    )

add_executable(algorithms_test_runner ${TEST_SOURCES})
target_link_libraries(algorithms_test_runner PRIVATE    algorithms_tests
                                                        lightning_compile_options
                                                        lightning_external_libs
                                                        )
catch_discover_tests(algorithms_test_runner)

install(TARGETS algorithms_test_runner DESTINATION bin)

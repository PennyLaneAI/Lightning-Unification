cmake_minimum_required(VERSION 3.20)

project(base_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${CMAKE_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define compile options and library
################################################################################

add_library(base_tests INTERFACE)
target_link_libraries(base_tests INTERFACE  Catch2::Catch2
                                            lightning_base
                                            lightning_utils
                                            )

target_link_libraries(base_tests INTERFACE   ${PL_BACKEND})

ProcessTestOptions(base_tests)

target_sources(base_tests INTERFACE runner_base.cpp)

################################################################################
# Define targets
################################################################################

# add_executable(compile_time_tests compile_time_tests.cpp)
# target_link_libraries(compile_time_tests lightning_compile_options lightning_gates lightning_utils lightning_simulator)

set(TEST_SOURCES    Test_StateVectorBase.cpp
                    )

add_executable(base_test_runner ${TEST_SOURCES})
target_link_libraries(base_test_runner PRIVATE  lightning_compile_options
                                                lightning_external_libs
                                                base_tests
                                                )
catch_discover_tests(base_test_runner)

# We build compile time tests before the runtime tests as build error messages
# are horrible if compile time constants are not well defined.
# add_dependencies(base_test_runner compile_time_tests)

install(TARGETS base_test_runner DESTINATION bin)

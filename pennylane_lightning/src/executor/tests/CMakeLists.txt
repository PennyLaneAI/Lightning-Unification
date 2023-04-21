cmake_minimum_required(VERSION 3.20)

project(executor_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${CMAKE_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define compile options and library
################################################################################

add_library(executor_tests INTERFACE)
target_link_libraries(executor_tests INTERFACE  Catch2::Catch2
                                                lightning_utils
                                                lightning_executor
                                                )

ProcessTestOptions(executor_tests)

target_sources(executor_tests INTERFACE runner_executor.cpp)

################################################################################
# Define targets
################################################################################
set(TEST_SOURCES    Test_Executor.cpp
                    )

add_executable(executor_test_runner ${TEST_SOURCES})
target_link_libraries(executor_test_runner PRIVATE  executor_tests
                                                    lightning_compile_options
                                                    lightning_external_libs
                                                    )
catch_discover_tests(executor_test_runner)

install(TARGETS executor_test_runner DESTINATION bin)

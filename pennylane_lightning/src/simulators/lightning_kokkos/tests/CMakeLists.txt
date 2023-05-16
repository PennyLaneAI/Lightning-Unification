cmake_minimum_required(VERSION 3.20)

project(pl_kokkos_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${CMAKE_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(pl_kokkos_tests INTERFACE)
target_link_libraries(pl_kokkos_tests INTERFACE Catch2::Catch2
                                                lightning_kokkos
                                                )

ProcessTestOptions(pl_kokkos_tests)

target_sources(pl_kokkos_tests INTERFACE runner_pl_kokkos.cpp)

################################################################################
# Define targets
################################################################################

set(TEST_SOURCES Test_StateVectorLKokkos.cpp)

add_executable(pl_kokkos_test_runner ${TEST_SOURCES})
target_link_libraries(pl_kokkos_test_runner PRIVATE pl_kokkos_tests)

catch_discover_tests(pl_kokkos_test_runner)

install(TARGETS pl_kokkos_test_runner DESTINATION bin)

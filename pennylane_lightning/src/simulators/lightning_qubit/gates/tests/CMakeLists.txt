cmake_minimum_required(VERSION 3.20)

project(pl_qubit_gates_tests)

# Default build type for test code is Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

include("${CMAKE_SOURCE_DIR}/cmake/support_tests.cmake")
FetchAndIncludeCatch()

################################################################################
# Define library
################################################################################

add_library(pl_qubit_gates_tests INTERFACE)
target_link_libraries(pl_qubit_gates_tests INTERFACE    Catch2::Catch2
                                                        pl_qubit_gates
                                                        )

ProcessTestOptions(pl_qubit_gates_tests)

target_sources(pl_qubit_gates_tests INTERFACE runner_pl_qubit_gates.cpp)

################################################################################
# Define targets
################################################################################

add_executable(compile_time_tests compile_time_tests.cpp)
target_link_libraries(compile_time_tests    lightning_compile_options
                                            pl_qubit_gates
                                            )

set(TEST_SOURCES    Test_AVXSingleQubitGateHelpers.cpp
                    Test_AVXTwoQubitGateHelpers.cpp
                    Test_DynamicDispatcher.cpp
                    Test_GateImplementations_CompareKernels.cpp
                    Test_GateImplementations_Generator.cpp
                    Test_GateImplementations_Inverse.cpp
                    Test_GateImplementations_Matrix.cpp
                    Test_GateImplementations_Nonparam.cpp
                    Test_GateImplementations_Param.cpp
                    Test_GateIndices.cpp
                    Test_Internal.cpp
                    Test_KernelMap.cpp
                    Test_OpToMemberFuncPtr.cpp
                    )

add_executable(pl_qubit_gates_test_runner ${TEST_SOURCES})
target_link_libraries(pl_qubit_gates_test_runner PRIVATE    pl_qubit_gates_tests)
catch_discover_tests(pl_qubit_gates_test_runner)

# We build compile time tests before the runtime tests as build error messages
# are horrible if compile time constants are not well defined.
add_dependencies(pl_qubit_gates_test_runner compile_time_tests)

install(TARGETS pl_qubit_gates_test_runner DESTINATION bin)

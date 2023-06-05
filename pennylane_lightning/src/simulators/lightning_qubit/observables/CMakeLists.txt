cmake_minimum_required(VERSION 3.20)

project(pl_qubit_observables LANGUAGES CXX)

set(OBSERVABLES_FILES ObservablesLQubit.cpp CACHE INTERNAL "" FORCE)
add_library(pl_qubit_observables STATIC ${OBSERVABLES_FILES})

target_include_directories(pl_qubit_observables INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pl_qubit_observables PRIVATE  lightning_compile_options
                                                    lightning_external_libs
                                                    )

target_link_libraries(pl_qubit_observables PUBLIC   lightning_utils
                                                    pl_qubit_utils
                                                    pl_qubit_gates
                                                    lightning_observables
                                                    lightning_qubit
                                                    )

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
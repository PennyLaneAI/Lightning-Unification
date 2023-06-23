cmake_minimum_required(VERSION 3.20)

project(lightning_observables LANGUAGES CXX)

add_library(lightning_observables INTERFACE ${OBSERVABLES_FILES})

target_include_directories(lightning_observables INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_observables INTERFACE   lightning_compile_options
                                                        lightning_external_libs
                                                        lightning_utils
                                                        )
if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
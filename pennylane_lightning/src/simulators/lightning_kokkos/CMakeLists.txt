cmake_minimum_required(VERSION 3.20)

set(LOGO [=[
░█░░░▀█▀░█▀▀░█░█░▀█▀░█▀█░▀█▀░█▀█░█▀▀░░░░█░█░█▀█░█░█░█░█░█▀█░█▀▀
░█░░░░█░░█░█░█▀█░░█░░█░█░░█░░█░█░█░█░░░░█▀▄░█░█░█▀▄░█▀▄░█░█░▀▀█
░▀▀▀░▀▀▀░▀▀▀░▀░▀░░▀░░▀░▀░▀▀▀░▀░▀░▀▀▀░▀░░▀░▀░▀▀▀░▀░▀░▀░▀░▀▀▀░▀▀▀
]=])
message(${LOGO})

project(lightning_kokkos
        DESCRIPTION "PennyLane Lightning Backend based in the Kokkos library."
        LANGUAGES CXX C
)

set(LKOKKOS_FILES StateVectorLKokkos.cpp CACHE INTERNAL "" FORCE)
add_library(lightning_kokkos STATIC ${LKOKKOS_FILES})

# Inform the compiler that this device is enabled.
target_compile_options(lightning_compile_options INTERFACE "-D_ENABLE_PLKOKKOS=1")

# Include macro and functions supporting Kokkos libraries.
include("${CMAKE_SOURCE_DIR}/cmake/support_kokkos.cmake")

FindKokkos(lightning_external_libs)

target_link_libraries(lightning_kokkos PUBLIC   lightning_compile_options
                                                lightning_external_libs
                                                lightning_base
                                                lightning_utils
                                                )

target_include_directories(lightning_kokkos PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET lightning_kokkos PROPERTY POSITION_INDEPENDENT_CODE ON)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()

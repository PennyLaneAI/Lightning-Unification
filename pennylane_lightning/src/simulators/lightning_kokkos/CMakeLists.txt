cmake_minimum_required(VERSION 3.20)
set(LOGO [=[
░█░░░▀█▀░█▀▀░█░█░▀█▀░█▀█░▀█▀░█▀█░█▀▀░░░░█░█░█▀█░█░█░█░█░█▀█░█▀▀
░█░░░░█░░█░█░█▀█░░█░░█░█░░█░░█░█░█░█░░░░█▀▄░█░█░█▀▄░█▀▄░█░█░▀▀█
░▀▀▀░▀▀▀░▀▀▀░▀░▀░░▀░░▀░▀░▀▀▀░▀░▀░▀▀▀░▀░░▀░▀░▀▀▀░▀░▀░▀░▀░▀▀▀░▀▀▀
]=])
message(${LOGO})

project(${PL_BACKEND}
    DESCRIPTION "PennyLane Lightning Kokkos C++ Backend."
    LANGUAGES CXX C
)

set(LKOKKOS_FILES   StateVectorKokkos.cpp
                    CACHE INTERNAL "" FORCE)

add_library(${PL_BACKEND} STATIC ${LKOKKOS_FILES})
target_compile_options(lightning_compile_options INTERFACE "-D_ENABLE_PLKOKKOS=1")

##########################
## Enforce C++ Standard ##
##########################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Ensure the libraries can see additional libs at same level;
# Required for external deps when loading in Python
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_SKIP_BUILD_RPATH FALSE)

#########################
## Set Default Options ##
#########################
option(PLKOKKOS_ENABLE_CLANG_TIDY "Enable clang-tidy build checks" OFF)
option(PLKOKKOS_ENABLE_NATIVE "Enable native CPU build tuning" OFF)
option(PLKOKKOS_ENABLE_SANITIZER "Enable address sanitizer" OFF)
option(PLKOKKOS_ENABLE_WARNINGS "Enable warnings" ON)



# Include macro and functions supporting Kokkos libraries.
include("${CMAKE_SOURCE_DIR}/cmake/support_kokkos.cmake")
FindKokkosAndKokkosKernels(lightning_external_libs)

if(PLKOKKOS_ENABLE_CLANG_TIDY)
    if (NOT DEFINED CLANG_TIDY_BINARY)
        set(CLANG_TIDY_BINARY clang-tidy)
    endif()
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_BINARY};
                            -extra-arg=-std=c++20;
    )
endif()

if(PLKOKKOS_ENABLE_SANITIZER)
    add_compile_options(-fsanitize=address)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_link_options(-fsanitize=address)
    else ()
        add_link_options(-fsanitize=address -static-libasan)
    endif()
    set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
ENDIF()

target_link_libraries(${PL_BACKEND} PUBLIC   lightning_compile_options
                                                lightning_external_libs
                                                lightning_base
                                                lightning_utils
                                                ${PL_BACKEND}_utils
                                                ${PL_BACKEND}_gates
                                                )
target_include_directories(${PL_BACKEND} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET ${PL_BACKEND} PROPERTY POSITION_INDEPENDENT_CODE ON)
if(PLKOKKOS_ENABLE_NATIVE)
    target_compile_options(${PL_BACKEND} PRIVATE -march=native)
endif()

###############################################################################
# Include subdirectories
###############################################################################
set(COMPONENT_SUBDIRS   algorithms
                        bindings
                        gates
                        measurements
                        observables
                        utils
                        )

foreach(COMP ${COMPONENT_SUBDIRS})
    add_subdirectory(${COMP})
endforeach()

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()

# ############################
# ## Create libraries
# ############################
# add_subdirectory(pennylane_lightning_kokkos/src)
# add_library(pennylane_lightning_kokkos INTERFACE)
# target_link_libraries(pennylane_lightning_kokkos INTERFACE ${PL_BACKEND}_utils lightning_kokkos_simulator lightning_kokkos_algorithms kokkos)
# target_include_directories(pennylane_lightning_kokkos INTERFACE "pennylane_lightning_kokkos/src")
# target_include_directories(pennylane_lightning_kokkos INTERFACE ${PLKOKKOS_KOKKOS_INC_DIR} )

# # Create binding module
# if(PLKOKKOS_ENABLE_PYTHON)
# include(FetchContent)
# FetchContent_Declare(
#     pybind11
#     GIT_REPOSITORY https://github.com/pybind/pybind11.git
#     GIT_TAG        v2.10.1
# )
# FetchContent_MakeAvailable(pybind11)

# find_package (Python COMPONENTS Interpreter Development)
#     if(Kokkos_ENABLE_CUDA)
#         add_library(lightning_kokkos_qubit_ops MODULE "pennylane_lightning_kokkos/src/bindings/Bindings.cpp")
#         target_link_libraries(lightning_kokkos_qubit_ops PUBLIC pybind11::module)
#         pybind11_extension(lightning_kokkos_qubit_ops)
#     else()
#         pybind11_add_module(lightning_kokkos_qubit_ops "pennylane_lightning_kokkos/src/bindings/Bindings.cpp" )
#     endif()
#     target_link_libraries(lightning_kokkos_qubit_ops PRIVATE pennylane_lightning_kokkos)
#     set_target_properties(lightning_kokkos_qubit_ops PROPERTIES CXX_VISIBILITY_PRESET "hidden" CUDA_VISIBILITY_PRESET "hidden")
#     target_compile_options(lightning_kokkos_qubit_ops PRIVATE "$<$<CONFIG:RELEASE>:-W>")
#     target_include_directories(lightning_kokkos_qubit_ops PRIVATE ${PLKOKKOS_KOKKOS_INC_DIR})
#     target_compile_definitions(lightning_kokkos_qubit_ops PRIVATE VERSION_INFO=${VERSION_STRING})
#     target_link_libraries(lightning_kokkos_qubit_ops PRIVATE kokkos)
# endif()


# if(PLKOKKOS_ENABLE_WARNINGS)
#     target_compile_options(pennylane_lightning_kokkos INTERFACE
#         $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror;>
#     )
#     if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#         target_compile_options(pennylane_lightning_kokkos INTERFACE
#             $<$<COMPILE_LANGUAGE:CXX>:-Wno-error=volatile;-Wno-error=deprecated-declarations>
#         )
#     endif()
#     # Explicitly disable warnings for Kokkos-specific library issues.
# endif()

# if(PLKOKKOS_ENABLE_NATIVE)
#     message(STATUS "ENABLE_NATIVE is ON. Using -march=native")
#     add_compile_options(-march=native)
#     target_compile_options(pennylane_lightning_kokkos INTERFACE -march=native)
#     target_compile_options(lightning_kokkos_qubit_ops PRIVATE -march=native)
# endif()

# install(TARGETS pennylane_lightning_kokkos
#         RUNTIME DESTINATION bin
#         LIBRARY DESTINATION lib
#         ARCHIVE DESTINATION lib
#         INCLUDES DESTINATION include
#         PUBLIC_HEADER DESTINATION include
# )

# install(DIRECTORY 
#     ${PROJECT_SOURCE_DIR}/pennylane_lightning_kokkos/src
#     DESTINATION include/pennylane_lightning_kokkos
# )

# if (PLKOKKOS_BUILD_TESTS)
#     enable_testing()
# endif()


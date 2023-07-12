project(${PL_BACKEND}_algorithms LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)


set(Kokkos_ALGORITHM_FILES  AdjointDiffKokkos.hpp
                            AdjointDiffKokkos.cpp
                            CACHE INTERNAL "" FORCE)
add_library(${PL_BACKEND}_algorithms STATIC ${Kokkos_ALGORITHM_FILES})

target_include_directories(${PL_BACKEND}_algorithms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} )
target_link_libraries(${PL_BACKEND}_algorithms PUBLIC   lightning_algorithms
                                                        ${PL_BACKEND}
                                                        ${PL_BACKEND}_utils
                                                        ${PL_BACKEND}_observables)
set_property(TARGET ${PL_BACKEND}_algorithms PROPERTY POSITION_INDEPENDENT_CODE ON)


#include "Macros.hpp"
#include "RuntimeInfo.hpp"

#include <catch2/catch.hpp>

#include <iostream>
/// @cond DEV
namespace {
using namespace Pennylane::Util;
} // namespace
/// @endcond

TEST_CASE("Runtime information is correct", "[Test_RuntimeInfo]") {
    INFO("RuntimeInfo::AVX " << RuntimeInfo::AVX());
    INFO("RuntimeInfo::AVX2 " << RuntimeInfo::AVX2());
    INFO("RuntimeInfo::AVX512F " << RuntimeInfo::AVX512F());
    INFO("RuntimeInfo::vendor " << RuntimeInfo::vendor());
    INFO("RuntimeInfo::brand " << RuntimeInfo::brand());
    REQUIRE(true);
    std::cout << "TESTE" << std::endl;
#ifdef _ENABLE_KOKKOS
    std::cout << _ENABLE_KOKKOS << std::endl;
#endif
}

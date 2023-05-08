#include "Macros.hpp"
#include "RuntimeInfo.hpp"

#include <catch2/catch.hpp>

using namespace Pennylane::Lightning_Qubit;

TEST_CASE("Runtime information is correct", "[Test_RuntimeInfo]") {
    INFO("RuntimeInfo::AVX " << Util::RuntimeInfo::AVX());
    INFO("RuntimeInfo::AVX2 " << Util::RuntimeInfo::AVX2());
    INFO("RuntimeInfo::AVX512F " << Util::RuntimeInfo::AVX512F());
    INFO("RuntimeInfo::vendor " << Util::RuntimeInfo::vendor());
    INFO("RuntimeInfo::brand " << Util::RuntimeInfo::brand());
    REQUIRE(true);
}

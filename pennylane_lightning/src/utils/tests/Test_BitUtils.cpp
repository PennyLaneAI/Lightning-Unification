#include "BitUtil.hpp"
#include "TestHelpers.hpp"
#include <catch2/catch.hpp>
#include <cmath>

using namespace Pennylane::Util;

TEMPLATE_TEST_CASE("BitUtil", "[BitUtil]", float, double) {
    SECTION("Check log2") { REQUIRE(log2(16) == log2PerfectPower(16)); }

    SECTION("Check isPerfectPowerOf2") {
        REQUIRE(isPerfectPowerOf2(16));
        REQUIRE(isPerfectPowerOf2(17) == false);
    }
}
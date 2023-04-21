#include <cmath>

#include "TestHelpers.hpp"
#include "Util.hpp"
#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE("Util:", "[Util]", float, double) {
    TestType ONE = 1;
    TestType TWO = 2;

    SECTION("Checking sqrt2") {
        REQUIRE(Pennylane::Util::SQRT2<TestType>() == std::sqrt(TWO));
    }

    SECTION("Checking inverted sqrt2") {
        REQUIRE(Pennylane::Util::INVSQRT2<TestType>() == ONE / std::sqrt(TWO));
    }

    SECTION("Checking exp2") {
        REQUIRE(Pennylane::Util::exp2(4) == std::exp2(4));
    }
}
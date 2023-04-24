#include "AdjointJacobian.hpp"
#include "TestHelpers.hpp"
#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE("AdjointJacobian::compare", "[Algorithms]", float, double) {
    SECTION("Compare numbers") {
        TestType number_a = 4;
        TestType number_b = 4;
        TestType number_c = 5;

        REQUIRE(number_a == number_b);
        REQUIRE(number_a != number_c);
    }
}
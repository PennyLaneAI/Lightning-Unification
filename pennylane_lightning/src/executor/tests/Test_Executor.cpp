#include "Executor.hpp"
#include "TestHelpers.hpp"
#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE("Executor::compare", "[Executor]", float, double) {
    SECTION("Compare numbers") {
        TestType number_a = 4;
        TestType number_b = 4;
        TestType number_c = 5;

        REQUIRE(number_a == number_b);
        REQUIRE(number_a != number_c);
    }
}
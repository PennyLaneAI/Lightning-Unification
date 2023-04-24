#include <complex>
#include <vector>

#include "StateVectorLQubit.hpp"
#include "TestHelpers.hpp"

#include <catch2/catch.hpp>

using namespace Pennylane;

TEMPLATE_TEST_CASE("LightningQubit::StateVectorLQubit::changeDataPtr",
                   "[StateVectorLQubit]", float, double) {
    using PrecisionT = TestType;
    using ComplexType = std::complex<TestType>;

    SECTION("changeDataPtr correctly update data") {
        std::vector<ComplexType> st_data(1 << 4);
        std::iota(st_data.begin(), st_data.end(), 0);
        StateVectorLQubit<PrecisionT> sv(st_data.data(), st_data.size());

        std::vector<ComplexType> st_data2(1 << 8);
        std::iota(st_data2.begin(), st_data2.end(), 1);
        sv.changeDataPtr(st_data2.data(), st_data2.size());

        REQUIRE(sv.getNumQubits() == 8);
        REQUIRE(sv.getData() == st_data2.data());
        REQUIRE(sv.getLength() == (1U << 8U));
    }

    SECTION("changeDataPtr throws an exception when the data is incorrect") {
        std::vector<ComplexType> st_data(1 << 4);
        std::iota(st_data.begin(), st_data.end(), 0);
        StateVectorLQubit<PrecisionT> sv(st_data.data(), st_data.size());

        std::vector<ComplexType> new_data(7);
        std::iota(new_data.begin(), new_data.end(), 0);

        REQUIRE_THROWS_AS(sv.changeDataPtr(new_data.data(), new_data.size()),
                          Util::LightningException);
    }
}

TEMPLATE_TEST_CASE("LightningQubit::StateVectorLQubit::setDataFrom",
                   "[StateVectorLQubit]", float, double) {
    using PrecisionT = TestType;
    using ComplexType = std::complex<TestType>;

    SECTION("setDataFrom correctly update data") {
        std::vector<ComplexType> st_data1(1 << 4);
        std::iota(st_data1.begin(), st_data1.end(), 0);

        std::vector<ComplexType> st_data2(1 << 4);
        std::iota(st_data2.begin(), st_data2.end(), 10);

        StateVectorLQubit<PrecisionT> sv(st_data1.data(), st_data1.size());

        sv.setDataFrom(st_data2.data(),
                       st_data2.size()); // Should update st_data1
        REQUIRE(st_data1 == st_data2);
    }

    SECTION("setDataFrom throws an exception when the data is incorrect") {
        std::vector<ComplexType> st_data1(1 << 4);
        std::iota(st_data1.begin(), st_data1.end(), 0);

        std::vector<ComplexType> st_data2(1 << 8);
        std::iota(st_data2.begin(), st_data2.end(), 10);

        StateVectorLQubit<PrecisionT> sv(st_data1.data(), st_data1.size());

        REQUIRE_THROWS_AS(sv.setDataFrom(st_data2.data(), st_data2.size()),
                          Util::LightningException);
    }
}
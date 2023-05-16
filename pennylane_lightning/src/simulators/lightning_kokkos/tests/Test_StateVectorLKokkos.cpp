#include <complex>
#include <vector>

#include "StateVectorLKokkos.hpp"
#include "TestHelpers.hpp"

#include <catch2/catch.hpp>

/// @cond DEV
namespace {
using namespace Pennylane;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("LightningKokkos::StateVectorLKokkos::Constructors",
                   "[StateVectorLKokkos]", float, double) {
    using PrecisionT = TestType;
    using ComplexType = std::complex<TestType>;

    SECTION("Initial view") {
        size_t num_qubits = 4;

        std::vector<ComplexType> st_data(1 << num_qubits);
        st_data[0] = 1;
        StateVectorLKokkos<PrecisionT> sv1(st_data.data(), st_data.size());

        StateVectorLKokkos<PrecisionT> sv2(num_qubits);

        REQUIRE(sv1 == sv2);
    }

    SECTION(
        "Constructor throws an exception when the array size is incorrect") {
        std::vector<ComplexType> new_data(7);
        std::iota(new_data.begin(), new_data.end(), 0);

        REQUIRE_THROWS_AS(
            StateVectorLKokkos<PrecisionT>(new_data.data(), new_data.size()),
            Util::LightningException);
    }
}

TEMPLATE_TEST_CASE("LightningKokkos::StateVectorLKokkos::updateData",
                   "[StateVectorLKokkos]", float, double) {
    using PrecisionT = TestType;
    using ComplexType = std::complex<TestType>;

    SECTION("updateData correctly update data") {
        size_t num_qubits = 4;

        std::vector<ComplexType> st_data(1 << num_qubits);
        st_data[0] = 11;
        StateVectorLKokkos<PrecisionT> sv1(st_data.data(), st_data.size());

        StateVectorLKokkos<PrecisionT> sv2(sv1);

        REQUIRE(sv1 == sv2);
    }
}

TEMPLATE_TEST_CASE("LightningKokkos::StateVectorLKokkos::resetStateVector",
                   "[StateVectorLKokkos]", float, double) {
    using PrecisionT = TestType;
    using ComplexType = std::complex<TestType>;

    SECTION("resetStateVector correctly resets the state") {
        size_t num_qubits = 4;

        std::vector<ComplexType> st_data(1 << num_qubits);
        st_data[0] = 11;
        StateVectorLKokkos<PrecisionT> sv1(st_data.data(), st_data.size());
        sv1.resetStateVector();

        StateVectorLKokkos<PrecisionT> sv2(num_qubits);

        REQUIRE(sv1 == sv2);
    }
}

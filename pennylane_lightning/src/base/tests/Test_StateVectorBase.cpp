#include <complex>
#include <vector>

#include <catch2/catch.hpp>

#include "TestHelpers.hpp"

#ifdef _ENABLE_PLQUBIT
#include "StateVectorLQubit.hpp"
template <typename T> struct StateVectorBackend {
    using StateVector = Pennylane::StateVectorLQubit<T>;
    using Precision = T;
};
#elif defined(_ENABLE_PLKOKKOS)
#include "StateVectorLKokkos.hpp"
template <typename T> struct StateVectorBackend {
    using StateVector = Pennylane::StateVectorLKokkos<T>;
    using Precision = T;
};
#endif

using namespace Pennylane;

TEMPLATE_PRODUCT_TEST_CASE("Base::StateVectorBase", "[StateVectorBase]",
                           (StateVectorBackend), (float, double)) {
    using StateVectorT = typename TestType::StateVector;
    using PrecisionT = typename TestType::Precision;
    using ComplexType = std::complex<PrecisionT>;

    SECTION("Testing StateVectorBase methods") {
        const size_t num_qubits = 4;

        std::vector<ComplexType> st_data1(1 << num_qubits);
        std::iota(st_data1.begin(), st_data1.end(), 0);
        StateVectorT sv1(st_data1.data(), st_data1.size());

        std::vector<ComplexType> st_data2(1 << num_qubits);
        std::iota(st_data2.begin(), st_data2.end(), 0);
        StateVectorT sv2(st_data2.data(), st_data2.size());

        REQUIRE(sv1.getNumQubits() == 4);
        REQUIRE(sv1.getLength() == 16);
    }
}

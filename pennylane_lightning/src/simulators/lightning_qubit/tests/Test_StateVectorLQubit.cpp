#include "LQubitTestHelpers.hpp" // createRandomStateVectorData
#include "LinearAlgebra.hpp"     //randomUnitary
#include "StateVectorLQubitManaged.hpp"
#include "StateVectorLQubitRaw.hpp"
#include "TestHelpers.hpp"
#include "cpu_kernels/GateImplementationsPI.hpp"

#include <algorithm>
#include <complex>
#include <random>
#include <type_traits>
#include <vector>

#include <catch2/catch.hpp>

/**
 * @file
 *  Tests for functionality:
 *      - defined in the intermediate base class StateVectorLQubit.
 *      - shared between all child classes.
 */

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit;
using namespace Pennylane::Util;

using Pennylane::LightningQubit::Util::createRandomStateVectorData;
using Pennylane::LightningQubit::Util::randomUnitary;
using Pennylane::LightningQubit::Util::TestVector;

std::mt19937_64 re{1337};
} // namespace
/// @endcond

template <typename T> struct StateVectorManagedAndPrecision {
    using StateVector = Pennylane::LightningQubit::StateVectorLQubitManaged<T>;
    using Precision = T;
};

template <typename T> struct StateVectorRawAndPrecision {
    using StateVector = Pennylane::LightningQubit::StateVectorLQubitRaw<T>;
    using Precision = T;
};

TEMPLATE_TEST_CASE("StateVectorLQubit::Constructibility",
                   "[Default Constructibility]", StateVectorLQubitRaw<>,
                   StateVectorLQubitManaged<>) {

    SECTION("StateVectorBackend<>") {
        REQUIRE(!std::is_constructible_v<TestType>);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("StateVectorLQubit::Constructibility",
                           "[General Constructibility]",
                           (StateVectorManagedAndPrecision,
                            StateVectorRawAndPrecision),
                           (float, double)) {
    using StateVectorT = typename TestType::StateVector;
    using PrecisionT = typename TestType::Precision;
    using ComplexT = std::complex<PrecisionT>;

    SECTION("StateVectorBackend<TestType>") {
        REQUIRE(!std::is_constructible_v<StateVectorT>);
    }
    SECTION("StateVectorBackend<TestType> {ComplexT*, size_t}") {
        using VectorT = TestVector<ComplexT>;
        REQUIRE(std::is_constructible_v<StateVectorT, ComplexT *, size_t>);

        const size_t num_qubits = 4;
        VectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);

        StateVectorT state_vector(st_data.data(), st_data.size());

        REQUIRE(state_vector.getNumQubits() == 4);
        REQUIRE(state_vector.getLength() == 16);
        REQUIRE(isApproxEqual(st_data.data(), st_data.size(),
                              state_vector.getData(),
                              state_vector.getLength()));
    }
    SECTION("StateVectorBackend<TestType> {ComplexT*, size_t}: Fails if "
            "provided an inconsistent length.") {
        std::vector<ComplexT> st_data(14, 0.0);
        REQUIRE_THROWS_WITH(
            StateVectorT(st_data.data(), st_data.size()),
            Catch::Contains("The size of provided data must be a power of 2."));
    }
    SECTION(
        "StateVectorBackend<TestType> {const StateVectorBackend<TestType>&}") {
        REQUIRE(std::is_copy_constructible_v<StateVectorT>);
    }
    SECTION("StateVectorBackend<TestType> {StateVectorBackend<TestType>&&}") {
        REQUIRE(std::is_move_constructible_v<StateVectorT>);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("StateVectorLQubit::applyMatrix with a std::vector",
                           "[applyMatrix]",
                           (StateVectorManagedAndPrecision,
                            StateVectorRawAndPrecision),
                           (float, double)) {
    using StateVectorT = typename TestType::StateVector;
    using PrecisionT = typename TestType::Precision;
    using ComplexT = std::complex<PrecisionT>;
    using VectorT = TestVector<ComplexT>;

    SECTION("Test wrong matrix size") {
        std::vector<ComplexT> m(7, 0.0);
        const size_t num_qubits = 4;
        VectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);

        StateVectorT state_vector(st_data.data(), st_data.size());
        REQUIRE_THROWS_WITH(
            state_vector.applyMatrix(m, {0, 1}),
            Catch::Contains(
                "The size of matrix does not match with the given"));
    }

    SECTION("Test wrong number of wires") {
        std::vector<ComplexT> m(8, 0.0);
        const size_t num_qubits = 4;
        VectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);

        StateVectorT state_vector(st_data.data(), st_data.size());
        REQUIRE_THROWS_WITH(
            state_vector.applyMatrix(m, {0}),
            Catch::Contains(
                "The size of matrix does not match with the given"));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("StateVectorLQubit::applyMatrix with a pointer",
                           "[applyMatrix]",
                           (StateVectorManagedAndPrecision,
                            StateVectorRawAndPrecision),
                           (float, double)) {
    using StateVectorT = typename TestType::StateVector;
    using PrecisionT = typename TestType::Precision;
    using ComplexT = std::complex<PrecisionT>;
    using VectorT = TestVector<ComplexT>;

    SECTION("Test wrong matrix") {
        std::vector<ComplexT> m(8, 0.0);
        const size_t num_qubits = 4;
        VectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);

        StateVectorT state_vector(st_data.data(), st_data.size());
        REQUIRE_THROWS_WITH(state_vector.applyMatrix(m.data(), {}),
                            Catch::Contains("must be larger than 0"));
    }

    SECTION("Test with different number of wires") {
        const size_t num_qubits = 5;
        for (size_t num_wires = 1; num_wires < num_qubits; num_wires++) {

            VectorT st_data_1 =
                createRandomStateVectorData<PrecisionT>(re, num_qubits);
            VectorT st_data_2 = st_data_1;

            StateVectorT state_vector_1(st_data_1.data(), st_data_1.size());
            StateVectorT state_vector_2(st_data_2.data(), st_data_2.size());

            std::vector<size_t> wires(num_wires);
            std::iota(wires.begin(), wires.end(), 0);

            const auto m = randomUnitary<PrecisionT>(re, num_wires);

            state_vector_1.applyMatrix(m, wires);

            Gates::GateImplementationsPI::applyMultiQubitOp<PrecisionT>(
                state_vector_2.getData(), num_qubits, m.data(), wires, false);

            REQUIRE(isApproxEqual(
                state_vector_1.getData(), state_vector_1.getLength(),
                state_vector_2.getData(), state_vector_2.getLength()));
        }
    }
}

TEMPLATE_PRODUCT_TEST_CASE("StateVectorLQubit::applyOperations",
                           "[applyOperations]",
                           (StateVectorManagedAndPrecision,
                            StateVectorRawAndPrecision),
                           (float, double)) {
    using StateVectorT = typename TestType::StateVector;
    using PrecisionT = typename TestType::Precision;
    using ComplexT = std::complex<PrecisionT>;
    using VectorT = TestVector<ComplexT>;

    SECTION("Test invalid arguments without parameters") {
        const size_t num_qubits = 4;
        VectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);

        StateVectorT state_vector(st_data.data(), st_data.size());

        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperations({"PauliX", "PauliY"}, {{0}},
                                         {false, false}),
            LightningException, "must all be equal"); // invalid wires
        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperations({"PauliX", "PauliY"}, {{0}, {1}},
                                         {false}),
            LightningException, "must all be equal"); // invalid inverse
    }

    SECTION("applyOperations without parameters works as expected") {
        const size_t num_qubits = 3;
        StateVectorLQubitManaged<PrecisionT> sv1(num_qubits);

        VectorT st_data_1 =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);
        VectorT st_data_2 = st_data_1;

        StateVectorT state_vector_1(st_data_1.data(), st_data_1.size());
        StateVectorT state_vector_2(st_data_2.data(), st_data_2.size());

        state_vector_1.applyOperations({"PauliX", "PauliY"}, {{0}, {1}},
                                       {false, false});

        state_vector_2.applyOperation("PauliX", {0}, false);
        state_vector_2.applyOperation("PauliY", {1}, false);

        REQUIRE(isApproxEqual(
            state_vector_1.getData(), state_vector_1.getLength(),
            state_vector_2.getData(), state_vector_2.getLength()));
    }

    SECTION("Test invalid arguments with parameters") {
        const size_t num_qubits = 4;

        VectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);

        StateVectorT state_vector(st_data.data(), st_data.size());

        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperations({"RX", "RY"}, {{0}}, {false, false},
                                         {{0.0}, {0.0}}),
            LightningException, "must all be equal"); // invalid wires

        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperations({"RX", "RY"}, {{0}, {1}}, {false},
                                         {{0.0}, {0.0}}),
            LightningException, "must all be equal"); // invalid wires

        PL_REQUIRE_THROWS_MATCHES(
            state_vector.applyOperations({"RX", "RY"}, {{0}, {1}},
                                         {false, false}, {{0.0}}),
            LightningException, "must all be equal"); // invalid parameters
    }

    SECTION("applyOperations with params works as expected") {
        const size_t num_qubits = 3;

        VectorT st_data_1 =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);
        VectorT st_data_2 = st_data_1;

        StateVectorT state_vector_1(st_data_1.data(), st_data_1.size());
        StateVectorT state_vector_2(st_data_2.data(), st_data_2.size());

        state_vector_1.applyOperations({"RX", "RY"}, {{0}, {1}}, {false, false},
                                       {{0.1}, {0.2}});

        state_vector_2.applyOperation("RX", {0}, false, {0.1});
        state_vector_2.applyOperation("RY", {1}, false, {0.2});

        REQUIRE(isApproxEqual(
            state_vector_1.getData(), state_vector_1.getLength(),
            state_vector_2.getData(), state_vector_2.getLength()));
    }
}

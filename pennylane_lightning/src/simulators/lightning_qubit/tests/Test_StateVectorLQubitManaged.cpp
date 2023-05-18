#include "LinearAlgebra.hpp" //randomUnitary
#include "StateVectorLQubitManaged.hpp"
// #include "StateVectorRawCPU.hpp"
#include "LQubitTestHelpers.hpp" // createRandomState
#include "TestHelpers.hpp"
// #include "Util.hpp"
#include "cpu_kernels/GateImplementationsPI.hpp"

#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit;
using namespace Pennylane::Util;

using Pennylane::LightningQubit::Util::createRandomState;
using Pennylane::LightningQubit::Util::randomUnitary;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("StateVectorLQubitManaged::StateVectorLQubitManaged",
                   "[StateVectorLQubitManaged]", float, double) {
    using PrecisionT = TestType;

    SECTION("StateVectorLQubitManaged") {
        REQUIRE(!std::is_constructible_v<StateVectorLQubitManaged<>>);
    }
    SECTION("StateVectorLQubitManaged<TestType>") {
        REQUIRE(!std::is_constructible_v<StateVectorLQubitManaged<TestType>>);
    }
    SECTION("StateVectorLQubitManaged<TestType> {size_t}") {
        REQUIRE(std::is_constructible_v<StateVectorLQubitManaged<TestType>,
                                        size_t>);
        const size_t num_qubits = 4;
        StateVectorLQubitManaged<PrecisionT> sv(num_qubits);

        REQUIRE(sv.getNumQubits() == 4);
        REQUIRE(sv.getLength() == 16);
        REQUIRE(sv.getDataVector().size() == 16);
    }
    // SECTION("StateVectorLQubitManaged<TestType> {const "
    //         "StateVectorRawCPU<TestType>&}") {
    //     REQUIRE(std::is_constructible_v<StateVectorLQubitManaged<TestType>,
    //                                     const StateVectorRawCPU<TestType>
    //                                     &>);
    // }
    SECTION("StateVectorLQubitManaged<TestType> {const "
            "StateVectorLQubitManaged<TestType>&}") {
        REQUIRE(
            std::is_copy_constructible_v<StateVectorLQubitManaged<TestType>>);
    }
    SECTION("StateVectorLQubitManaged<TestType> "
            "{StateVectorLQubitManaged<TestType>&&}") {
        REQUIRE(
            std::is_move_constructible_v<StateVectorLQubitManaged<TestType>>);
    }
    SECTION("Aligned 256bit statevector") {
        const auto memory_model = CPUMemoryModel::Aligned256;
        StateVectorLQubitManaged<PrecisionT> sv(4, Threading::SingleThread,
                                                memory_model);
        /* Even when we allocate 256 bit aligned memory it is possible that the
         * alignment happens to be 512 bit */
        REQUIRE(((getMemoryModel(sv.getDataVector().data()) ==
                  CPUMemoryModel::Aligned256) ||
                 (getMemoryModel(sv.getDataVector().data()) ==
                  CPUMemoryModel::Aligned512)));
    }

    SECTION("Aligned 512bit statevector") {
        const auto memory_model = CPUMemoryModel::Aligned512;
        StateVectorLQubitManaged<PrecisionT> sv(4, Threading::SingleThread,
                                                memory_model);
        REQUIRE((getMemoryModel(sv.getDataVector().data()) ==
                 CPUMemoryModel::Aligned512));
    }
}

TEMPLATE_TEST_CASE("StateVectorLQubitManaged::applyMatrix with std::vector",
                   "[StateVectorLQubitManaged]", float, double) {
    using PrecisionT = TestType;
    SECTION("Test wrong matrix size") {
        std::vector<std::complex<TestType>> m(7, 0.0);
        const size_t num_qubits = 4;
        StateVectorLQubitManaged<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS_WITH(
            sv.applyMatrix(m, {0, 1}),
            Catch::Contains(
                "The size of matrix does not match with the given"));
    }

    SECTION("Test wrong number of wires") {
        std::vector<std::complex<TestType>> m(8, 0.0);
        const size_t num_qubits = 4;
        StateVectorLQubitManaged<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS_WITH(
            sv.applyMatrix(m, {0}),
            Catch::Contains(
                "The size of matrix does not match with the given"));
    }
}

TEMPLATE_TEST_CASE("StateVectorLQubitManaged::applyMatrix with a pointer",
                   "[StateVectorLQubitManaged]", float, double) {
    using PrecisionT = TestType;
    SECTION("Test wrong matrix") {
        std::vector<std::complex<TestType>> m(8, 0.0);
        const size_t num_qubits = 4;
        StateVectorLQubitManaged<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS_WITH(sv.applyMatrix(m.data(), {}),
                            Catch::Contains("must be larger than 0"));
    }

    SECTION("Test with different number of wires") {
        std::default_random_engine re{1337};
        const size_t num_qubits = 5;
        for (size_t num_wires = 1; num_wires < num_qubits; num_wires++) {
            StateVectorLQubitManaged<PrecisionT> sv1(num_qubits);
            StateVectorLQubitManaged<PrecisionT> sv2(num_qubits);

            std::vector<size_t> wires(num_wires);
            std::iota(wires.begin(), wires.end(), 0);

            const auto m = randomUnitary<PrecisionT>(re, num_wires);
            sv1.applyMatrix(m, wires);
            Gates::GateImplementationsPI::applyMultiQubitOp<PrecisionT>(
                sv2.getData(), num_qubits, m.data(), wires, false);
            REQUIRE(sv1.getDataVector() ==
                    approx(sv2.getDataVector()).margin(PrecisionT{1e-5}));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorLQubitManaged::applyOperations",
                   "[StateVectorLQubitManaged]", float, double) {
    using PrecisionT = TestType;

    std::mt19937 re{1337};

    SECTION("Test invalid arguments without params") {
        const size_t num_qubits = 4;
        StateVectorLQubitManaged<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS_WITH(
            sv.applyOperations({"PauliX", "PauliY"}, {{0}}, {false, false}),
            Catch::Contains("must all be equal")); // invalid wires
        REQUIRE_THROWS_WITH(
            sv.applyOperations({"PauliX", "PauliY"}, {{0}, {1}}, {false}),
            Catch::Contains("must all be equal")); // invalid inverse
    }

    SECTION("applyOperations without params works as expected") {
        const size_t num_qubits = 3;
        StateVectorLQubitManaged<PrecisionT> sv1(num_qubits);

        sv1.updateData(createRandomState<PrecisionT>(re, num_qubits));
        StateVectorLQubitManaged<PrecisionT> sv2 = sv1;

        sv1.applyOperations({"PauliX", "PauliY"}, {{0}, {1}}, {false, false});

        sv2.applyOperation("PauliX", {0}, false);
        sv2.applyOperation("PauliY", {1}, false);

        REQUIRE(sv1.getDataVector() == approx(sv2.getDataVector()));
    }

    SECTION("Test invalid arguments with params") {
        const size_t num_qubits = 4;
        StateVectorLQubitManaged<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS_WITH(
            sv.applyOperations({"RX", "RY"}, {{0}}, {false, false},
                               {{0.0}, {0.0}}),
            Catch::Contains("must all be equal")); // invalid wires
        REQUIRE_THROWS_WITH(
            sv.applyOperations({"RX", "RY"}, {{0}, {1}}, {false},
                               {{0.0}, {0.0}}),
            Catch::Contains("must all be equal")); // invalid inverse

        REQUIRE_THROWS_WITH(
            sv.applyOperations({"RX", "RY"}, {{0}, {1}}, {false, false},
                               {{0.0}}),
            Catch::Contains("must all be equal")); // invalid params
    }

    SECTION("applyOperations with params works as expected") {
        const size_t num_qubits = 3;
        StateVectorLQubitManaged<PrecisionT> sv1(num_qubits);

        sv1.updateData(createRandomState<PrecisionT>(re, num_qubits));
        StateVectorLQubitManaged<PrecisionT> sv2 = sv1;

        sv1.applyOperations({"RX", "RY"}, {{0}, {1}}, {false, false},
                            {{0.1}, {0.2}});

        sv2.applyOperation("RX", {0}, false, {0.1});
        sv2.applyOperation("RY", {1}, false, {0.2});

        REQUIRE(sv1.getDataVector() == approx(sv2.getDataVector()));
    }
}

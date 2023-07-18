#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "MeasurementsKokkos.hpp"
#include "ObservablesKokkos.hpp"
#include "StateVectorKokkos.hpp"

/**
 * @file
 *  Tests for functionality:
 *      - expectation values.
 */

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos::Measures;
using namespace Pennylane::LightningKokkos::Observables;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::getExpectationValueIdentity",
                   "[StateVectorKokkosManaged_Expval]", float, double) {
    const std::size_t num_qubits = 3;
    auto ONE = TestType(1);
    StateVectorKokkos<TestType> kokkos_sv{num_qubits};
    auto m = Measurements(kokkos_sv);

    SECTION("Apply directly") {
        kokkos_sv.applyOperation("Hadamard", {0}, false);
        kokkos_sv.applyOperation("CNOT", {0, 1}, false);
        kokkos_sv.applyOperation("CNOT", {1, 2}, false);
        auto res = m.getExpectationValueIdentity({0});
        CHECK(res == Approx(ONE));
    }

    SECTION("Using expval") {
        kokkos_sv.applyOperation("Hadamard", {0}, false);
        kokkos_sv.applyOperation("CNOT", {0, 1}, false);
        kokkos_sv.applyOperation("CNOT", {1, 2}, false);
        auto ob = NamedObs<StateVectorKokkos<TestType>>("Identity", {0});
        auto res = m.expval(ob);
        CHECK(res == Approx(ONE));
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::getExpectationValuePauliX",
                   "[StateVectorKokkosManaged_Expval]", float, double) {
    {
        const std::size_t num_qubits = 3;

        auto ZERO = TestType(0);
        auto ONE = TestType(1);

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);
            kokkos_sv.applyOperation("Hadamard", {0}, false);
            kokkos_sv.applyOperation("CNOT", {0, 1}, false);
            kokkos_sv.applyOperation("CNOT", {1, 2}, false);
            auto res = m.getExpectationValuePauliX({0});
            CHECK(res == ZERO);
        }
        SECTION("Using expval") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);
            kokkos_sv.applyOperation("Hadamard", {0}, false);
            kokkos_sv.applyOperation("CNOT", {0, 1}, false);
            kokkos_sv.applyOperation("CNOT", {1, 2}, false);
            auto ob = NamedObs<StateVectorKokkos<TestType>>("PauliX", {0});
            auto res = m.expval(ob);
            CHECK(res == ZERO);
        }
        SECTION("Apply directly: Plus states") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);
            kokkos_sv.applyOperation("Hadamard", {0}, false);
            kokkos_sv.applyOperation("Hadamard", {1}, false);
            kokkos_sv.applyOperation("Hadamard", {2}, false);
            auto res = m.getExpectationValuePauliX({0});
            CHECK(res == Approx(ONE));
        }
        SECTION("Using expval: Plus states") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);
            kokkos_sv.applyOperation("Hadamard", {0}, false);
            kokkos_sv.applyOperation("Hadamard", {1}, false);
            kokkos_sv.applyOperation("Hadamard", {2}, false);
            auto ob = NamedObs<StateVectorKokkos<TestType>>("PauliX", {0});
            auto res = m.expval(ob);
            CHECK(res == Approx(ONE));
        }
        SECTION("Apply directly: Minus states") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);
            kokkos_sv.applyPauliX({0}, false);
            kokkos_sv.applyOperation("Hadamard", {0}, false);
            kokkos_sv.applyPauliX({1}, false);
            kokkos_sv.applyOperation("Hadamard", {1}, false);
            kokkos_sv.applyPauliX({2}, false);
            kokkos_sv.applyOperation("Hadamard", {2}, false);
            auto res = m.getExpectationValuePauliX({0});
            CHECK(res == -Approx(ONE));
        }
        SECTION("Using expval: Minus states") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);
            kokkos_sv.applyPauliX({0}, false);
            kokkos_sv.applyOperation("Hadamard", {0}, false);
            kokkos_sv.applyPauliX({1}, false);
            kokkos_sv.applyOperation("Hadamard", {1}, false);
            kokkos_sv.applyPauliX({2}, false);
            kokkos_sv.applyOperation("Hadamard", {2}, false);
            auto ob = NamedObs<StateVectorKokkos<TestType>>("PauliX", {0});
            auto res = m.expval(ob);
            CHECK(res == -Approx(ONE));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::getExpectationValuePauliY",
                   "[StateVectorKokkosManaged_Expval]", float, double) {
    {
        const std::size_t num_qubits = 3;

        auto ZERO = TestType(0);
        auto ONE = TestType(1);
        auto PI = TestType(M_PI);

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);
            kokkos_sv.applyOperation("Hadamard", {0}, false);
            kokkos_sv.applyOperation("CNOT", {0, 1}, false);
            kokkos_sv.applyOperation("CNOT", {1, 2}, false);
            auto res = m.getExpectationValuePauliY({0});
            CHECK(res == ZERO);
        }
        SECTION("Using expval") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);
            kokkos_sv.applyOperation("Hadamard", {0}, false);
            kokkos_sv.applyOperation("CNOT", {0, 1}, false);
            kokkos_sv.applyOperation("CNOT", {1, 2}, false);
            auto ob = NamedObs<StateVectorKokkos<TestType>>("PauliY", {0});
            auto res = m.expval(ob);
            CHECK(res == ZERO);
        }
        SECTION("Apply directly: Plus i states") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);
            kokkos_sv.applyRX({0}, false, {-PI / 2});
            kokkos_sv.applyRX({1}, false, {-PI / 2});
            kokkos_sv.applyRX({2}, false, {-PI / 2});
            auto res = m.getExpectationValuePauliY({0});
            CHECK(res == Approx(ONE));
        }
        SECTION("Using expval: Plus i states") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);
            kokkos_sv.applyRX({0}, false, {-PI / 2});
            kokkos_sv.applyRX({1}, false, {-PI / 2});
            kokkos_sv.applyRX({2}, false, {-PI / 2});
            auto ob = NamedObs<StateVectorKokkos<TestType>>("PauliY", {0});
            auto res = m.expval(ob);
            CHECK(res == Approx(ONE));
        }
        SECTION("Apply directly: Minus i states") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);
            kokkos_sv.applyRX({0}, false, {PI / 2});
            kokkos_sv.applyRX({1}, false, {PI / 2});
            kokkos_sv.applyRX({2}, false, {PI / 2});
            auto res = m.getExpectationValuePauliY({0});
            CHECK(res == -Approx(ONE));
        }
        SECTION("Using expval: Minus i states") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);
            kokkos_sv.applyRX({0}, false, {PI / 2});
            kokkos_sv.applyRX({1}, false, {PI / 2});
            kokkos_sv.applyRX({2}, false, {PI / 2});
            auto ob = NamedObs<StateVectorKokkos<TestType>>("PauliY", {0});
            auto res = m.expval(ob);
            CHECK(res == -Approx(ONE));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::getExpectationValuePauliZ",
                   "[StateVectorKokkosManaged_Expval]", float, double) {
    {
        const std::size_t num_qubits = 3;

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);
            kokkos_sv.applyOperation("Hadamard", {0}, false);
            kokkos_sv.applyOperation("CNOT", {0, 1}, false);
            kokkos_sv.applyOperation("CNOT", {1, 2}, false);
            auto res = m.getExpectationValuePauliZ({0});
            CHECK(res == 0); // A 0-result is not a good test.
        }
        SECTION("Using expval") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);
            kokkos_sv.applyOperation("Hadamard", {0}, false);
            kokkos_sv.applyOperation("CNOT", {0, 1}, false);
            kokkos_sv.applyOperation("CNOT", {1, 2}, false);
            auto ob = NamedObs<StateVectorKokkos<TestType>>("PauliZ", {0});
            auto res = m.expval(ob);
            CHECK(res == 0); // A 0-result is not a good test.
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::getExpectationValueHadamard",
                   "[StateVectorKokkosManaged_Expval]", float, double) {
    {
        const std::size_t num_qubits = 3;
        auto INVSQRT2 = TestType(0.707106781186547524401);

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);
            auto res = m.getExpectationValueHadamard({0});
            CHECK(res == INVSQRT2);
        }
        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);
            kokkos_sv.applyPauliX({0});
            auto res = m.getExpectationValueHadamard({0});
            CHECK(res == -INVSQRT2);
        }
        SECTION("Using expval") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);
            kokkos_sv.applyPauliX({0});
            auto ob = NamedObs<StateVectorKokkos<TestType>>("Hadamard", {0});
            auto res = m.expval(ob);
            CHECK(res == -INVSQRT2);
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::getExpectationValueSingleQubitOp",
                   "[StateVectorKokkosManaged_Expval]", float, double) {
    {
        using ComplexT = StateVectorKokkos<TestType>::ComplexT;
        const std::size_t num_qubits = 3;

        auto INVSQRT2 = TestType(0.707106781186547524401);

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);
            Kokkos::View<ComplexT *> opMatDevice("opMat", 4);
            Kokkos::View<ComplexT *, Kokkos::HostSpace> opMat("opMatHost", 4);

            const TestType theta = M_PI / 2;
            const TestType c = std::cos(theta / 2);
            const TestType js = std::sin(-theta / 2);

            opMat[0] = c;
            opMat[1] = ComplexT(static_cast<TestType>(0), js);
            opMat[2] = ComplexT(static_cast<TestType>(0), js);
            opMat[3] = c;

            Kokkos::deep_copy(opMatDevice, opMat);
            auto res = m.getExpectationValueSingleQubitOp(opMatDevice, {0});
            CHECK(res == INVSQRT2);
        }
        SECTION("Using expval") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);

            const TestType theta = M_PI / 2;
            const TestType c = std::cos(theta / 2);
            const TestType js = std::sin(-theta / 2);
            std::vector<ComplexT> matrix{c, {0, js}, {0, js}, c};

            auto ob = HermitianObs<StateVectorKokkos<TestType>>(matrix, {0});
            auto res = m.expval(ob);
            CHECK(res == INVSQRT2);
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkosManaged::getExpectationValueTwoQubitOp",
                   "[StateVectorKokkosManaged_Expval]", float, double) {
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    {
        const std::size_t num_qubits = 3;
        auto INVSQRT2 = TestType(0.707106781186547524401);

        SECTION("Apply directly") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);

            Kokkos::View<ComplexT *> opMatDevice("opMat", 16);
            Kokkos::View<ComplexT *, Kokkos::HostSpace> opMat("opMatHost", 16);

            const TestType theta = M_PI / 2;
            const TestType c = std::cos(theta / 2);
            const TestType js = std::sin(-theta / 2);

            opMat[0] = c;
            opMat[1] = ComplexT(static_cast<TestType>(0), js);
            opMat[4] = ComplexT(static_cast<TestType>(0), js);
            opMat[5] = c;
            opMat[10] = ComplexT(static_cast<TestType>(1), 0);
            opMat[15] = ComplexT(static_cast<TestType>(1), 0);

            Kokkos::deep_copy(opMatDevice, opMat);
            auto res = m.getExpectationValueTwoQubitOp(opMatDevice, {0, 1});
            CHECK(res == INVSQRT2);
        }
        SECTION("Using expval") {
            StateVectorKokkos<TestType> kokkos_sv{num_qubits};
            auto m = Measurements(kokkos_sv);

            const TestType theta = M_PI / 2;
            const TestType c = std::cos(theta / 2);
            const TestType js = std::sin(-theta / 2);
            std::vector<ComplexT> matrix(16);
            matrix[0] = c;
            matrix[1] = ComplexT(0, js);
            matrix[4] = ComplexT(0, js);
            matrix[5] = c;
            matrix[10] = ComplexT(1, 0);
            matrix[15] = ComplexT(1, 0);

            auto ob = HermitianObs<StateVectorKokkos<TestType>>(matrix, {0, 1});
            auto res = m.expval(ob);
            CHECK(res == INVSQRT2);
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::Hamiltonian_expval",
                   "[StateVectorKokkos_Expval]", float, double) {
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    const std::size_t num_qubits = 3;
    SECTION("GetExpectationIdentity") {
        StateVectorKokkos<TestType> kokkos_sv{num_qubits};
        auto m = Measurements(kokkos_sv);
        std::vector<size_t> wires{0, 1, 2};

        kokkos_sv.applyHadamard({0}, false);
        kokkos_sv.applyCNOT({0, 1}, false);
        kokkos_sv.applyCNOT({1, 2}, false);

        size_t matrix_dim = static_cast<size_t>(1U) << num_qubits;
        std::vector<ComplexT> matrix(matrix_dim * matrix_dim);

        for (size_t i = 0; i < matrix.size(); i++) {
            if (i % matrix_dim == i / matrix_dim)
                matrix[i] = ComplexT(1, 0);
            else
                matrix[i] = ComplexT(0, 0);
        }

        auto results = m.getExpectationValue(wires, matrix);
        ComplexT expected = {1, 0};
        CHECK(real(expected) == Approx(results).epsilon(1e-7));
    }

    SECTION("GetExpectationHermitianMatrix") {
        std::vector<ComplexT> init_state{{0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1},
                                         {0.1, 0.2}, {0.2, 0.2}, {0.3, 0.3},
                                         {0.3, 0.4}, {0.4, 0.5}};
        StateVectorKokkos<TestType> kokkos_sv{init_state.data(),
                                              init_state.size()};
        auto m = Measurements(kokkos_sv);
        std::vector<size_t> wires{0, 1, 2};
        std::vector<ComplexT> matrix{
            {0.5, 0.0},  {0.2, 0.5},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5},
            {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},
            {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5},
            {0.3, 0.0},  {0.5, 0.0},  {0.2, 0.5},  {0.2, -0.5}, {0.3, 0.0},
            {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5},
            {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},
            {0.2, -0.5}, {0.3, 0.0},  {0.5, 0.0},  {0.2, 0.5},  {0.2, -0.5},
            {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},
            {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5},
            {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.5, 0.0},  {0.2, 0.5},
            {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5},
            {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},
            {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0}};

        auto results = m.getExpectationValue(wires, matrix);
        ComplexT expected(1.263000, -1.011000);
        CHECK(real(expected) == Approx(results).epsilon(1e-7));
    }

    SECTION("Using expval") {
        std::vector<ComplexT> init_state{{0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1},
                                         {0.1, 0.2}, {0.2, 0.2}, {0.3, 0.3},
                                         {0.3, 0.4}, {0.4, 0.5}};
        StateVectorKokkos<TestType> kokkos_sv{init_state.data(),
                                              init_state.size()};
        std::vector<ComplexT> matrix{
            {0.5, 0.0},  {0.2, 0.5},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5},
            {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},
            {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5},
            {0.3, 0.0},  {0.5, 0.0},  {0.2, 0.5},  {0.2, -0.5}, {0.3, 0.0},
            {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5},
            {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},
            {0.2, -0.5}, {0.3, 0.0},  {0.5, 0.0},  {0.2, 0.5},  {0.2, -0.5},
            {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},
            {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5},
            {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.5, 0.0},  {0.2, 0.5},
            {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5},
            {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0},
            {0.2, -0.5}, {0.3, 0.0},  {0.2, -0.5}, {0.3, 0.0}};

        auto m = Measurements(kokkos_sv);
        auto ob = HermitianObs<StateVectorKokkos<TestType>>(matrix, {0, 1, 2});
        auto res = m.expval(ob);
        ComplexT expected(1.263000, -1.011000);
        CHECK(real(expected) == Approx(res).epsilon(1e-7));
    }
}

TEMPLATE_TEST_CASE("Test expectation value of HamiltonianObs",
                   "[StateVectorKokkos_Expval]", float, double) {
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    SECTION("Using expval") {
        std::vector<ComplexT> init_state{{0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1},
                                         {0.1, 0.2}, {0.2, 0.2}, {0.3, 0.3},
                                         {0.3, 0.4}, {0.4, 0.5}};
        StateVectorKokkos<TestType> kokkos_sv{init_state.data(),
                                              init_state.size()};
        auto m = Measurements(kokkos_sv);

        auto X0 = std::make_shared<NamedObs<StateVectorKokkos<TestType>>>(
            "PauliX", std::vector<size_t>{0});
        auto Z1 = std::make_shared<NamedObs<StateVectorKokkos<TestType>>>(
            "PauliZ", std::vector<size_t>{1});

        auto ob = Hamiltonian<StateVectorKokkos<TestType>>::create({0.3, 0.5},
                                                                   {X0, Z1});
        auto res = m.expval(*ob);
        auto expected = TestType(-0.086);
        CHECK(expected == Approx(res));
    }
}

TEMPLATE_TEST_CASE("Test expectation value of TensorProdObs",
                   "[StateVectorKokkos_Expval]", float, double) {
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    SECTION("Using expval") {
        std::vector<ComplexT> init_state{{0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1},
                                         {0.1, 0.2}, {0.2, 0.2}, {0.3, 0.3},
                                         {0.3, 0.4}, {0.4, 0.5}};
        StateVectorKokkos<TestType> kokkos_sv{init_state.data(),
                                              init_state.size()};
        auto m = Measurements(kokkos_sv);

        auto X0 = std::make_shared<NamedObs<StateVectorKokkos<TestType>>>(
            "PauliX", std::vector<size_t>{0});
        auto Z1 = std::make_shared<NamedObs<StateVectorKokkos<TestType>>>(
            "PauliZ", std::vector<size_t>{1});

        auto ob = TensorProdObs<StateVectorKokkos<TestType>>::create({X0, Z1});
        auto res = m.expval(*ob);
        auto expected = TestType(-0.36);
        CHECK(expected == Approx(res));
    }
}

TEMPLATE_TEST_CASE("StateVectorKokkos::Hamiltonian_expval_Sparse",
                   "[StateVectorKokkos_Expval]", float, double) {
    using ComplexT = StateVectorKokkos<TestType>::ComplexT;
    SECTION("GetExpectationSparse") {
        std::vector<ComplexT> init_state{{0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1},
                                         {0.1, 0.2}, {0.2, 0.2}, {0.3, 0.3},
                                         {0.3, 0.4}, {0.4, 0.5}};
        StateVectorKokkos<TestType> kokkos_sv{init_state.data(),
                                              init_state.size()};
        auto m = Measurements(kokkos_sv);

        std::vector<size_t> index_ptr = {0, 2, 4, 6, 8, 10, 12, 14, 16};
        std::vector<size_t> indices = {0, 3, 1, 2, 1, 2, 0, 3,
                                       4, 7, 5, 6, 5, 6, 4, 7};
        std::vector<ComplexT> values = {
            {3.1415, 0.0},  {0.0, -3.1415}, {3.1415, 0.0}, {0.0, 3.1415},
            {0.0, -3.1415}, {3.1415, 0.0},  {0.0, 3.1415}, {3.1415, 0.0},
            {3.1415, 0.0},  {0.0, -3.1415}, {3.1415, 0.0}, {0.0, 3.1415},
            {0.0, -3.1415}, {3.1415, 0.0},  {0.0, 3.1415}, {3.1415, 0.0}};

        auto result = m.getExpectationValue(values, indices, index_ptr);
        auto expected = TestType(3.1415);
        CHECK(expected == Approx(result).epsilon(1e-7));
    }

    SECTION("Sparse expval") {
        std::vector<ComplexT> init_state{{0.0, 0.0}, {0.0, 0.1}, {0.1, 0.1},
                                         {0.1, 0.2}, {0.2, 0.2}, {0.3, 0.3},
                                         {0.3, 0.4}, {0.4, 0.5}};
        StateVectorKokkos<TestType> kokkos_sv{init_state.data(),
                                              init_state.size()};
        auto m = Measurements(kokkos_sv);

        std::vector<size_t> index_ptr = {0, 2, 4, 6, 8, 10, 12, 14, 16};
        std::vector<size_t> indices = {0, 3, 1, 2, 1, 2, 0, 3,
                                       4, 7, 5, 6, 5, 6, 4, 7};
        std::vector<ComplexT> values = {
            {3.1415, 0.0},  {0.0, -3.1415}, {3.1415, 0.0}, {0.0, 3.1415},
            {0.0, -3.1415}, {3.1415, 0.0},  {0.0, 3.1415}, {3.1415, 0.0},
            {3.1415, 0.0},  {0.0, -3.1415}, {3.1415, 0.0}, {0.0, 3.1415},
            {0.0, -3.1415}, {3.1415, 0.0},  {0.0, 3.1415}, {3.1415, 0.0}};

        auto result = m.expval(index_ptr.data(), index_ptr.size(),
                               indices.data(), values.data(), values.size());
        auto expected = TestType(3.1415);
        CHECK(expected == Approx(result).epsilon(1e-7));
    }
}
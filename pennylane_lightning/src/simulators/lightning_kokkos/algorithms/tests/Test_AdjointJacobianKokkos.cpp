// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <catch2/catch.hpp>

#include "AdjointJacobianKokkos.hpp"
#include "StateVectorKokkos.hpp"
#include "TestHelpers.hpp"
#include "Util.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Util;
using namespace Pennylane::LightningKokkos::Algorithms;
using Pennylane::Algorithms::OpsData;
} // namespace
/// @endcond

/**
 * @brief Tests the constructability of the AdjointDiff.hpp classes.
 *
 */
TEMPLATE_TEST_CASE("AdjointJacobian::AdjointJacobian", "[AdjointJacobian]",
                   float, double) {
    SECTION("AdjointJacobian<TestType> {}") {
        REQUIRE(std::is_constructible<
                AdjointJacobian<StateVectorKokkos<TestType>>>::value);
    }
}

TEST_CASE("AdjointJacobian::AdjointJacobian Op=RX, Obs=Z",
          "[AdjointJacobian]") {
    using StateVectorT = StateVectorKokkos<double>;

    AdjointJacobian<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};

    const auto num_threads_2 =
        Kokkos::InitializationSettings().set_num_threads(2);
    const size_t num_qubits = 1;
    const size_t num_params = 3;
    const size_t num_obs = 1;
    auto obs = std::make_shared<NamedObs<StateVectorT>>("PauliZ",
                                                        std::vector<size_t>{0});
    std::vector<std::vector<double>> jacobian(
        num_obs, std::vector<double>(num_params, 0));
    std::vector<StateVectorT> st_vecs = {
        StateVectorT{num_qubits}, StateVectorT{num_qubits, num_threads_2}};
    for (auto &psi : st_vecs) {
        for (const auto &p : param) {
            auto ops = OpsData<StateVectorT>{{"RX"}, {{p}}, {{0}}, {false}};
            psi.resetStateVector();
            adj.adjointJacobian(psi, jacobian, {obs}, ops, {0}, true);
            CHECK(-sin(p) == Approx(jacobian[0][0]).margin(1e-5));
        }
    }
}

TEST_CASE("AdjointJacobian::adjointJacobian Op=RY, Obs=X",
          "[AdjointJacobian]") {
    using StateVectorT = StateVectorKokkos<double>;

    AdjointJacobian<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    const size_t num_qubits = 1;
    const size_t num_params = 3;
    const size_t num_obs = 1;

    auto obs = std::make_shared<NamedObs<StateVectorT>>("PauliX",
                                                        std::vector<size_t>{0});
    std::vector<std::vector<double>> jacobian(
        num_obs, std::vector<double>(num_params, 0));
    StateVectorT psi(num_qubits);
    for (const auto &p : param) {
        auto ops = OpsData<StateVectorT>{{"RY"}, {{p}}, {{0}}, {false}};
        psi.resetStateVector();
        adj.adjointJacobian(psi, jacobian, {obs}, ops, {0}, true);
        CHECK(cos(p) == Approx(jacobian[0][0]).margin(1e-5));
    }
}

TEST_CASE("AdjointJacobian::adjointJacobian Op=RX, Obs=[Z,Z]",
          "[AdjointJacobian]") {
    using StateVectorT = StateVectorKokkos<double>;
    AdjointJacobian<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    {
        const size_t num_qubits = 2;
        const size_t num_params = 1;
        const size_t num_obs = 2;
        std::vector<std::vector<double>> jacobian(
            num_obs, std::vector<double>(num_params, 0));

        StateVectorT psi(num_qubits);

        auto obs1 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{0});
        auto obs2 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{1});

        auto ops = OpsData<StateVectorT>{{"RX"}, {{param[0]}}, {{0}}, {false}};

        adj.adjointJacobian(psi, jacobian, {obs1, obs2}, ops, {0}, true);

        CAPTURE(jacobian);
        CHECK(-sin(param[0]) == Approx(jacobian[0][0]).margin(1e-7));
        CHECK(0.0 == Approx(jacobian[1][0]).margin(1e-7));
    }
}

TEST_CASE("AdjointJacobian::adjointJacobian Op=[RX,RX,RX], Obs=[Z,Z,Z]",
          "[AdjointJacobian]") {
    using StateVectorT = StateVectorKokkos<double>;
    AdjointJacobian<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    {
        const size_t num_qubits = 3;
        const size_t num_params = 3;
        const size_t num_obs = 3;
        std::vector<std::vector<double>> jacobian(
            num_obs, std::vector<double>(num_params, 0));

        StateVectorT psi(num_qubits);

        auto obs1 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{0});
        auto obs2 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{1});
        auto obs3 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{2});

        auto ops = OpsData<StateVectorT>{{"RX", "RX", "RX"},
                                     {{param[0]}, {param[1]}, {param[2]}},
                                     {{0}, {1}, {2}}, {false, false, false}};

        adj.adjointJacobian(psi, jacobian, {obs1, obs2, obs3}, ops, {0, 1, 2},
                            true);

        CAPTURE(jacobian);
        CHECK(-sin(param[0]) == Approx(jacobian[0][0]).margin(1e-7));
        CHECK(-sin(param[1]) == Approx(jacobian[1][1]).margin(1e-7));
        CHECK(-sin(param[2]) == Approx(jacobian[2][2]).margin(1e-7));
    }
}

TEST_CASE("AdjointJacobian::adjointJacobian Op=[RX,RX,RX], Obs=[Z,Z,Z],"
          "TParams=[0,2]",
          "[AdjointJacobian]") {
    using StateVectorT = StateVectorKokkos<double>;
    AdjointJacobian<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    {
        const size_t num_qubits = 3;
        const size_t num_params = 3;
        const size_t num_obs = 3;
        std::vector<std::vector<double>> jacobian(
            num_obs, std::vector<double>(num_params, 0));
        std::vector<size_t> t_params{0, 2};

        StateVectorT psi(num_qubits);

        auto obs1 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{0});
        auto obs2 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{1});
        auto obs3 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{2});

        auto ops = OpsData<StateVectorT>{{"RX", "RX", "RX"},
                                     {{param[0]}, {param[1]}, {param[2]}},
                                     {{0}, {1}, {2}}, {false, false, false}};

        adj.adjointJacobian(psi, jacobian, {obs1, obs2, obs3}, ops, t_params,
                            true);

        CAPTURE(jacobian);
        CHECK(-sin(param[0]) == Approx(jacobian[0][0]).margin(1e-7));
        CHECK(0 == Approx(jacobian[1][1]).margin(1e-7));
        CHECK(-sin(param[2]) == Approx(jacobian[2][1]).margin(1e-7));
    }
}

TEST_CASE("AdjointJacobian::adjointJacobian Op=[RX,RX,RX], Obs=[ZZZ]",
          "[AdjointJacobian]") {
    using StateVectorT = StateVectorKokkos<double>;
    AdjointJacobian<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    {
        const size_t num_qubits = 3;
        const size_t num_params = 3;
        const size_t num_obs = 1;
        std::vector<std::vector<double>> jacobian(
            num_obs, std::vector<double>(num_params, 0));

        StateVectorT psi(num_qubits);

        auto obs = std::make_shared<TensorProdObs<StateVectorT>>(
            std::make_shared<NamedObs<StateVectorT>>("PauliZ",
                                                     std::vector<size_t>{0}),
            std::make_shared<NamedObs<StateVectorT>>("PauliZ",
                                                     std::vector<size_t>{1}),
            std::make_shared<NamedObs<StateVectorT>>("PauliZ",
                                                     std::vector<size_t>{2}));
        auto ops = OpsData<StateVectorT>{{"RX", "RX", "RX"},
                                     {{param[0]}, {param[1]}, {param[2]}},
                                     {{0}, {1}, {2}}, {false, false, false}};

        adj.adjointJacobian(psi, jacobian, {obs}, ops, {0, 1, 2}, true);
        CAPTURE(jacobian);

        // Computed with parameter shift
        CHECK(-0.1755096592645253 == Approx(jacobian[0][0]).margin(1e-7));
        CHECK(0.26478810666384334 == Approx(jacobian[0][1]).margin(1e-7));
        CHECK(-0.6312451595102775 == Approx(jacobian[0][2]).margin(1e-7));
    }
}

TEST_CASE("AdjointJacobian::adjointJacobian Op=Mixed, Obs=[XXX]",
          "[AdjointJacobian]") {
    using StateVectorT = StateVectorKokkos<double>;
    AdjointJacobian<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    {
        const size_t num_qubits = 3;
        const size_t num_params = 6;
        const size_t num_obs = 1;
        std::vector<std::vector<double>> jacobian(
            num_obs, std::vector<double>(num_params, 0));

        StateVectorT psi(num_qubits);

        auto obs = std::make_shared<TensorProdObs<StateVectorT>>(
            std::make_shared<NamedObs<StateVectorT>>("PauliX",
                                                     std::vector<size_t>{0}),
            std::make_shared<NamedObs<StateVectorT>>("PauliX",
                                                     std::vector<size_t>{1}),
            std::make_shared<NamedObs<StateVectorT>>("PauliX",
                                                     std::vector<size_t>{2}));

        auto ops = OpsData<StateVectorT>{
            {"RZ", "RY", "RZ", "CNOT", "CNOT", "RZ", "RY", "RZ"},
            {{param[0]},
             {param[1]},
             {param[2]},
             {},
             {},
             {param[0]},
             {param[1]},
             {param[2]}},
            {{0}, {0}, {0}, {0, 1}, {1, 2}, {1}, {1}, {1}},
            {false, false, false, false, false, false, false, false}};

        adj.adjointJacobian(psi, jacobian, {obs}, ops, {0, 1, 2, 3, 4, 5},
                            true);
        CAPTURE(jacobian);

        // Computed with PennyLane using default.qubit.adjoint_jacobian
        CHECK(0.0 == Approx(jacobian[0][0]).margin(1e-7));
        CHECK(-0.674214427 == Approx(jacobian[0][1]).margin(1e-7));
        CHECK(0.275139672 == Approx(jacobian[0][2]).margin(1e-7));
        CHECK(0.275139672 == Approx(jacobian[0][3]).margin(1e-7));
        CHECK(-0.0129093062 == Approx(jacobian[0][4]).margin(1e-7));
        CHECK(0.323846156 == Approx(jacobian[0][5]).margin(1e-7));
    }
}

TEST_CASE("AdjointJacobian::adjointJacobian Decomposed Rot gate, non "
          "computational basis state",
          "[AdjointJacobian]") {
    using StateVectorT = StateVectorKokkos<double>;
    AdjointJacobian<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    {
        const size_t num_params = 3;
        const size_t num_obs = 1;

        const auto thetas = linspace(-2 * M_PI, 2 * M_PI, 7);
        std::unordered_map<double, std::vector<double>> expec_results{
            {thetas[0], {0, -9.90819496e-01, 0}},
            {thetas[1], {-8.18996553e-01, 1.62526544e-01, 0}},
            {thetas[2], {-0.203949, 0.48593716, 0}},
            {thetas[3], {0, 1, 0}},
            {thetas[4], {-2.03948985e-01, 4.85937177e-01, 0}},
            {thetas[5], {-8.18996598e-01, 1.62526487e-01, 0}},
            {thetas[6], {0, -9.90819511e-01, 0}}};

        for (const auto &theta : thetas) {
            std::vector<double> local_params{theta, std::pow(theta, 3),
                                             M_SQRT2 * theta};
            std::vector<std::vector<double>> jacobian(
                num_obs, std::vector<double>(num_params, 0));

            std::vector<Kokkos::complex<double>> cdata{
                {INVSQRT2<Kokkos::complex, double>()},
                {-INVSQRT2<Kokkos::complex, double>()}};
            std::vector<Kokkos::complex<double>> new_data{cdata.begin(),
                                                          cdata.end()};
            StateVectorT psi(new_data.data(), new_data.size());

            auto obs = std::make_shared<NamedObs<StateVectorT>>(
                "PauliZ", std::vector<size_t>{0});
            auto ops = OpsData<StateVectorT>{
                {"RZ", "RY", "RZ"},
                {{local_params[0]}, {local_params[1]}, {local_params[2]}},
                {{0}, {0}, {0}}, {false, false, false}};

            adj.adjointJacobian(psi, jacobian, {obs}, ops, {0, 1, 2}, true);
            CAPTURE(theta);
            CAPTURE(jacobian);

            // Computed with PennyLane using default.qubit
            CHECK(expec_results[theta][0] ==
                  Approx(jacobian[0][0]).margin(1e-7));
            CHECK(expec_results[theta][1] ==
                  Approx(jacobian[0][1]).margin(1e-7));
            CHECK(expec_results[theta][2] ==
                  Approx(jacobian[0][2]).margin(1e-7));
        }
    }
}

TEST_CASE("AdjointJacobian::adjointJacobian Mixed Ops, Obs and TParams",
          "[AdjointJacobian]") {
    using StateVectorT = StateVectorKokkos<double>;
    AdjointJacobian<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    {
        const std::vector<size_t> t_params{1, 2, 3};
        const size_t num_obs = 1;

        const auto thetas = linspace(-2 * M_PI, 2 * M_PI, 8);

        std::vector<double> local_params{0.543, 0.54, 0.1,  0.5, 1.3,
                                         -2.3,  0.5,  -0.5, 0.5};
        std::vector<std::vector<double>> jacobian(
            num_obs, std::vector<double>(t_params.size(), 0));

        std::vector<Kokkos::complex<double>> cdata{
            {ONE<Kokkos::complex, double>()},
            {ZERO<Kokkos::complex, double>()},
            {ZERO<Kokkos::complex, double>()},
            {ZERO<Kokkos::complex, double>()}};
        std::vector<Kokkos::complex<double>> new_data{cdata.begin(),
                                                      cdata.end()};
        StateVectorT psi(new_data.data(), new_data.size());

        const auto obs = std::make_shared<TensorProdObs<StateVectorT>>(
            std::make_shared<NamedObs<StateVectorT>>("PauliX",
                                                     std::vector<size_t>{0}),
            std::make_shared<NamedObs<StateVectorT>>("PauliZ",
                                                     std::vector<size_t>{1}));
        auto ops =
            OpsData<StateVectorT>{{"Hadamard", "RX", "CNOT", "RZ", "RY", "RZ", "RZ",
                               "RY", "RZ", "RZ", "RY", "CNOT"},
                              {{},
                               {local_params[0]},
                               {},
                               {local_params[1]},
                               {local_params[2]},
                               {local_params[3]},
                               {local_params[4]},
                               {local_params[5]},
                               {local_params[6]},
                               {local_params[7]},
                               {local_params[8]},
                               {}},
                              std::vector<std::vector<std::size_t>>{{0},
                                                                    {0},
                                                                    {0, 1},
                                                                    {0},
                                                                    {0},
                                                                    {0},
                                                                    {0},
                                                                    {0},
                                                                    {0},
                                                                    {0},
                                                                    {1},
                                                                    {0, 1}},
                              {false, false, false, false, false, false, false,
                               false, false, false, false, false}};

        adj.adjointJacobian(psi, jacobian, {obs}, ops, t_params, true);

        std::vector<double> expected{-0.71429188, 0.04998561, -0.71904837};
        // Computed with PennyLane using default.qubit
        CHECK(expected[0] == Approx(jacobian[0][0]));
        CHECK(expected[1] == Approx(jacobian[0][1]));
        CHECK(expected[2] == Approx(jacobian[0][2]));
    }
}

TEST_CASE("Algorithms::adjointJacobian Op=RX, Obs=Ham[Z0+Z1]", "[Algorithms]") {
    using StateVectorT = StateVectorKokkos<double>;
    AdjointJacobian<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> tp{0};
    {
        const size_t num_qubits = 2;
        const size_t num_obs = 1;
        std::vector<std::vector<double>> jacobian(
            num_obs, std::vector<double>(tp.size(), 0));

        StateVectorT psi(num_qubits);

        const auto obs1 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{0});
        const auto obs2 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{1});

        auto ham = Hamiltonian<StateVectorT>::create({0.3, 0.7}, {obs1, obs2});

        auto ops = OpsData<StateVectorT>{
            {"RX"}, {{param[0]}}, {{0}}, {false}};

        adj.adjointJacobian(psi, jacobian, {ham}, ops, tp, true);

        CAPTURE(jacobian);
        CHECK(-0.3 * sin(param[0]) == Approx(jacobian[0][0]).margin(1e-7));
    }
}

TEST_CASE("AdjointJacobian::AdjointJacobian Op=[RX,RX,RX], "
          "Obs=Ham[Z0+Z1+Z2], "
          "TParams=[0,2]",
          "[AdjointJacobian]") {
    using StateVectorT = StateVectorKokkos<double>;
    AdjointJacobian<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> t_params{0, 2};
    {
        const size_t num_qubits = 3;
        const size_t num_obs = 1;
        std::vector<std::vector<double>> jacobian(
            num_obs, std::vector<double>(t_params.size(), 0));

        StateVectorT psi(num_qubits);

        auto obs1 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{0});
        auto obs2 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{1});
        auto obs3 = std::make_shared<NamedObs<StateVectorT>>(
            "PauliZ", std::vector<size_t>{2});

        auto ham = Hamiltonian<StateVectorT>::create({0.47, 0.32, 0.96},
                                                     {obs1, obs2, obs3});

        auto ops = OpsData<StateVectorT>{{"RX", "RX", "RX"},
                                     {{param[0]}, {param[1]}, {param[2]}},
                                     {{0}, {1}, {2}}, {false, false, false}};

        adj.adjointJacobian(psi, jacobian, {ham}, ops, t_params, true);
        CAPTURE(jacobian);

        CHECK((-0.47 * sin(param[0]) == Approx(jacobian[0][0]).margin(1e-7)));
        CHECK((-0.96 * sin(param[2]) == Approx(jacobian[0][1]).margin(1e-7)));
    }
}

TEST_CASE("AdjointJacobian::AdjointJacobian Test HermitianObs",
          "[AdjointJacobian]") {
    using StateVectorT = StateVectorKokkos<double>;
    using ComplexT = typename StateVectorT::ComplexT;
    AdjointJacobian<StateVectorT> adj;
    std::vector<double> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> t_params{0, 2};
    {
        const size_t num_qubits = 3;
        const size_t num_obs = 1;

        std::vector<std::vector<double>> jacobian1(
            num_obs, std::vector<double>(t_params.size(), 0));
        std::vector<std::vector<double>> jacobian2(
            num_obs, std::vector<double>(t_params.size(), 0));

        StateVectorT psi(num_qubits);

        auto obs1 = std::make_shared<TensorProdObs<StateVectorT>>(
            std::make_shared<NamedObs<StateVectorT>>("PauliZ",
                                                     std::vector<size_t>{0}),
            std::make_shared<NamedObs<StateVectorT>>("PauliZ",
                                                     std::vector<size_t>{1}));
        auto obs2 = std::make_shared<HermitianObs<StateVectorT>>(
            std::vector<ComplexT>{1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0,
                                  1},
            std::vector<size_t>{0, 1});

        auto ops = OpsData<StateVectorT>{{"RX", "RX", "RX"},
                                     {{param[0]}, {param[1]}, {param[2]}},
                                     {{0}, {1}, {2}}, {false, false, false}};

        adj.adjointJacobian(psi, jacobian1, {obs1}, ops, t_params, true);
        adj.adjointJacobian(psi, jacobian2, {obs2}, ops, t_params, true);

        CHECK((jacobian1[0] == PLApprox(jacobian2[0]).margin(1e-7)));
    }
}

#include "Error.hpp" // LightningException
#include "Observables.hpp"
#include "TestHelpers.hpp" // isApproxEqual, createZeroState, createProductState
#include "TypeList.hpp"
#include "Util.hpp" // TestVector

#include <catch2/catch.hpp>

#include <complex>
#include <memory>
#include <vector>
/**
 * @file
 *  Tests for Base Observable classes.
 */

/// @cond DEV
namespace {
using namespace Pennylane::Observables;

using Pennylane::Util::createProductState;
using Pennylane::Util::createZeroState;
using Pennylane::Util::isApproxEqual;
using Pennylane::Util::LightningException;
using Pennylane::Util::TestVector;
} // namespace
/// @endcond

#ifdef _ENABLE_PLQUBIT
constexpr bool BACKEND_FOUND = true;

#include "TestHelpersStateVectors.hpp" // TestStateVectorBackends, StateVectorToName

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit::Util;
} // namespace
/// @endcond

#elif _ENABLE_PLKOKKOS == 1
constexpr bool BACKEND_FOUND = true;

#include "TestHelpersStateVectors.hpp" // TestStateVectorBackends, StateVectorToName

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos::Util;
} // namespace
  /// @endcond

#else
constexpr bool BACKEND_FOUND = false;
using TestStateVectorBackends = Pennylane::Util::TypeList<void>;

template <class StateVector> struct StateVectorToName {};
#endif

template <typename TypeList> void testNamedObsBase() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using NamedObsT = NamedObsBase<StateVectorT>;

        DYNAMIC_SECTION("Name of the Observable must be correct - "
                        << StateVectorToName<StateVectorT>::name) {
            REQUIRE(NamedObsT("PauliZ", {0}).getObsName() == "PauliZ[0]");
        }

        DYNAMIC_SECTION("Comparing objects names") {
            auto ob1 = NamedObsT("PauliX", {0});
            auto ob2 = NamedObsT("PauliX", {0});
            auto ob3 = NamedObsT("PauliZ", {0});

            REQUIRE(ob1 == ob2);
            REQUIRE(ob2 != ob3);
            REQUIRE(ob1 != ob3);
        }

        DYNAMIC_SECTION("Comparing objects wires") {
            auto ob1 = NamedObsT("PauliY", {0});
            auto ob2 = NamedObsT("PauliY", {0});
            auto ob3 = NamedObsT("PauliY", {1});

            REQUIRE(ob1 == ob2);
            REQUIRE(ob2 != ob3);
            REQUIRE(ob1 != ob3);
        }

        DYNAMIC_SECTION("Comparing objects parameters") {
            auto ob1 = NamedObsT("RZ", {0}, {0.4});
            auto ob2 = NamedObsT("RZ", {0}, {0.4});
            auto ob3 = NamedObsT("RZ", {0}, {0.1});

            REQUIRE(ob1 == ob2);
            REQUIRE(ob2 != ob3);
            REQUIRE(ob1 != ob3);
        }

        testNamedObsBase<typename TypeList::Next>();
    }
}

TEST_CASE("Methods implemented in the NamedObsBase class", "[NamedObsBase]") {
    if constexpr (BACKEND_FOUND) {
        testNamedObsBase<TestStateVectorBackends>();
    }
}

template <typename TypeList> void testHermitianObsBase() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using ComplexT = typename StateVectorT::ComplexT;
        using HermitianObsT = HermitianObsBase<StateVectorT>;

        DYNAMIC_SECTION("HermitianObs only accepts correct arguments - "
                        << StateVectorToName<StateVectorT>::name) {
            auto ob1 =
                HermitianObsT{std::vector<ComplexT>{0.0, 0.0, 0.0, 0.0}, {0}};
            auto ob2 =
                HermitianObsT{std::vector<ComplexT>(16, ComplexT{}), {0, 1}};
            REQUIRE_THROWS_AS(
                HermitianObsT(std::vector<ComplexT>{0.0, 0.0, 0.0}, {0}),
                LightningException);
            REQUIRE_THROWS_AS(
                HermitianObsT(std::vector<ComplexT>{0.0, 0.0, 0.0, 0.0, 0.0},
                              {0, 1}),
                LightningException);
        }

        DYNAMIC_SECTION("getObsName - "
                        << StateVectorToName<StateVectorT>::name) {
            REQUIRE(
                HermitianObsT(std::vector<ComplexT>{1.0, 0.0, 2.0, 0.0}, {0})
                    .getObsName() == "Hermitian");
        }

        DYNAMIC_SECTION("Comparing objects matrices - "
                        << StateVectorToName<StateVectorT>::name) {
            auto ob1 =
                HermitianObsT{std::vector<ComplexT>{1.0, 0.0, 0.0, 0.0}, {0}};
            auto ob2 =
                HermitianObsT{std::vector<ComplexT>{1.0, 0.0, 0.0, 0.0}, {0}};
            auto ob3 =
                HermitianObsT{std::vector<ComplexT>{0.0, 1.0, 0.0, 0.0}, {0}};
            REQUIRE(ob1 == ob2);
            REQUIRE(ob1 != ob3);
            REQUIRE(ob2 != ob3);
        }

        DYNAMIC_SECTION("Comparing objects wires - "
                        << StateVectorToName<StateVectorT>::name) {
            auto ob1 =
                HermitianObsT{std::vector<ComplexT>{1.0, 0.0, -1.0, 0.0}, {0}};
            auto ob2 =
                HermitianObsT{std::vector<ComplexT>{1.0, 0.0, -1.0, 0.0}, {0}};
            auto ob3 =
                HermitianObsT{std::vector<ComplexT>{1.0, 0.0, -1.0, 0.0}, {1}};
            REQUIRE(ob1 == ob2);
            REQUIRE(ob1 != ob3);
            REQUIRE(ob2 != ob3);
        }

        testHermitianObsBase<typename TypeList::Next>();
    }
}

TEST_CASE("Methods implemented in the HermitianObsBase class",
          "[HermitianObsBase]") {
    if constexpr (BACKEND_FOUND) {
        testHermitianObsBase<TestStateVectorBackends>();
    }
}

template <typename TypeList> void testTensorProdObsBase() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;
        using ComplexT = typename StateVectorT::ComplexT;
        using HermitianObsT = HermitianObsBase<StateVectorT>;
        using NamedObsT = NamedObsBase<StateVectorT>;
        using TensorProdObsT = TensorProdObsBase<StateVectorT>;

        DYNAMIC_SECTION("Overlapping wires throw an exception - "
                        << StateVectorToName<StateVectorT>::name) {
            auto ob1 = std::make_shared<HermitianObsT>(
                std::vector<ComplexT>(16, ComplexT{0.0, 0.0}),
                std::vector<size_t>{0, 1});
            auto ob2_1 =
                std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{1});
            auto ob2_2 =
                std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{2});
            auto ob2 = TensorProdObsT::create({ob2_1, ob2_2});

            REQUIRE_THROWS_AS(TensorProdObsT::create({ob1, ob2}),
                              LightningException);
        }

        DYNAMIC_SECTION(
            "Constructing an observable with non-overlapping wires - "
            << StateVectorToName<StateVectorT>::name) {
            auto ob1 = std::make_shared<HermitianObsT>(
                std::vector<ComplexT>(16, ComplexT{0.0, 0.0}),
                std::vector<size_t>{0, 1});
            auto ob2_1 =
                std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{2});
            auto ob2_2 =
                std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{3});
            auto ob2 = TensorProdObsT::create({ob2_1, ob2_2});

            REQUIRE_NOTHROW(TensorProdObsT::create({ob1, ob2}));
        }

        DYNAMIC_SECTION("getObsName - "
                        << StateVectorToName<StateVectorT>::name) {
            auto ob = TensorProdObsT(
                std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{0}),
                std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{1}));
            REQUIRE(ob.getObsName() == "PauliX[0] @ PauliZ[1]");
        }

        DYNAMIC_SECTION("Compare tensor product observables"
                        << StateVectorToName<StateVectorT>::name) {
            auto ob1 = TensorProdObsT{
                std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{0}),
                std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{1})};
            auto ob2 = TensorProdObsT{
                std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{0}),
                std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{1})};
            auto ob3 = TensorProdObsT{
                std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{0}),
                std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{2})};
            auto ob4 = TensorProdObsT{
                std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{0}),
                std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{1})};

            auto ob5 = TensorProdObsT{
                std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{0})};

            REQUIRE(ob1 == ob2);
            REQUIRE(ob1 != ob3);
            REQUIRE(ob1 != ob4);
            REQUIRE(ob1 != ob5);
        }

        DYNAMIC_SECTION("Tensor product applies to a statevector correctly"
                        << StateVectorToName<StateVectorT>::name) {
            using VectorT = TestVector<ComplexT>;

            auto obs = TensorProdObsT{
                std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{0}),
                std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{2}),
            };

            SECTION("Test using |1+0>") {
                VectorT st_data =
                    createProductState<PrecisionT, ComplexT>("1+0");

                StateVectorT state_vector(st_data.data(), st_data.size());

                obs.applyInPlace(state_vector);

                VectorT expected =
                    createProductState<PrecisionT, ComplexT>("0+1");

                REQUIRE(isApproxEqual(state_vector.getData(),state_vector.getLength(), expected.data(), expected.size()));
            }

            SECTION("Test using |+-01>") {
                VectorT st_data =
                    createProductState<PrecisionT, ComplexT>("+-01");

                StateVectorT state_vector(st_data.data(), st_data.size());

                obs.applyInPlace(state_vector);

                VectorT expected =
                    createProductState<PrecisionT, ComplexT>("+-11");

                REQUIRE(isApproxEqual(state_vector.getData(),state_vector.getLength(), expected.data(), expected.size()));
            }
        }

        testTensorProdObsBase<typename TypeList::Next>();
    }
}

TEST_CASE("Methods implemented in the TensorProdObsBase class",
          "[TensorProdObsBase]") {
    if constexpr (BACKEND_FOUND) {
        testTensorProdObsBase<TestStateVectorBackends>();
    }
}

template <typename TypeList> void testHamiltonianBase() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;
        using ComplexT = typename StateVectorT::ComplexT;
        using NamedObsT = NamedObsBase<StateVectorT>;
        using TensorProdObsT = TensorProdObsBase<StateVectorT>;
        using HamiltonianT = HamiltonianBase<StateVectorT>;

        const auto h = PrecisionT{0.809}; // half of the golden ratio

        auto zz = std::make_shared<TensorProdObsT>(
            std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{0}),
            std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{1}));

        auto x1 = std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{0});
        auto x2 = std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{1});

        DYNAMIC_SECTION(
            "Hamiltonian constructor only accepts valid arguments - "
            << StateVectorToName<StateVectorT>::name) {
            REQUIRE_NOTHROW(
                HamiltonianT::create({PrecisionT{1.0}, h, h}, {zz, x1, x2}));

            REQUIRE_THROWS_AS(
                HamiltonianT::create({PrecisionT{1.0}, h}, {zz, x1, x2}),
                LightningException);

            DYNAMIC_SECTION("getObsName - "
                            << StateVectorToName<StateVectorT>::name) {
                auto X0 = std::make_shared<NamedObsT>("PauliX",
                                                      std::vector<size_t>{0});
                auto Z2 = std::make_shared<NamedObsT>("PauliZ",
                                                      std::vector<size_t>{2});

                REQUIRE(
                    HamiltonianT::create({0.3, 0.5}, {X0, Z2})->getObsName() ==
                    "Hamiltonian: { 'coeffs' : [0.3, 0.5], "
                    "'observables' : [PauliX[0], PauliZ[2]]}");
            }

            DYNAMIC_SECTION("Compare Hamiltonians - "
                            << StateVectorToName<StateVectorT>::name) {
                auto X0 = std::make_shared<NamedObsT>("PauliX",
                                                      std::vector<size_t>{0});
                auto X1 = std::make_shared<NamedObsT>("PauliX",
                                                      std::vector<size_t>{1});
                auto X2 = std::make_shared<NamedObsT>("PauliX",
                                                      std::vector<size_t>{2});

                auto Y0 = std::make_shared<NamedObsT>("PauliY",
                                                      std::vector<size_t>{0});
                auto Y1 = std::make_shared<NamedObsT>("PauliY",
                                                      std::vector<size_t>{1});
                auto Y2 = std::make_shared<NamedObsT>("PauliY",
                                                      std::vector<size_t>{2});

                auto Z0 = std::make_shared<NamedObsT>("PauliZ",
                                                      std::vector<size_t>{0});
                auto Z1 = std::make_shared<NamedObsT>("PauliZ",
                                                      std::vector<size_t>{1});
                auto Z2 = std::make_shared<NamedObsT>("PauliZ",
                                                      std::vector<size_t>{2});

                auto ham1 = HamiltonianT::create(
                    {0.8, 0.5, 0.7},
                    {
                        std::make_shared<TensorProdObsT>(X0, Y1, Z2),
                        std::make_shared<TensorProdObsT>(Z0, X1, Y2),
                        std::make_shared<TensorProdObsT>(Y0, Z1, X2),
                    });

                auto ham2 = HamiltonianT::create(
                    {0.8, 0.5, 0.7},
                    {
                        std::make_shared<TensorProdObsT>(X0, Y1, Z2),
                        std::make_shared<TensorProdObsT>(Z0, X1, Y2),
                        std::make_shared<TensorProdObsT>(Y0, Z1, X2),
                    });

                auto ham3 = HamiltonianT::create(
                    {0.8, 0.5, 0.642},
                    {
                        std::make_shared<TensorProdObsT>(X0, Y1, Z2),
                        std::make_shared<TensorProdObsT>(Z0, X1, Y2),
                        std::make_shared<TensorProdObsT>(Y0, Z1, X2),
                    });

                auto ham4 = HamiltonianT::create(
                    {0.8, 0.5},
                    {
                        std::make_shared<TensorProdObsT>(X0, Y1, Z2),
                        std::make_shared<TensorProdObsT>(Z0, X1, Y2),
                    });

                auto ham5 = HamiltonianT::create(
                    {0.8, 0.5, 0.7},
                    {
                        std::make_shared<TensorProdObsT>(X0, Y1, Z2),
                        std::make_shared<TensorProdObsT>(Z0, X1, Y2),
                        std::make_shared<TensorProdObsT>(Y0, Z1, Y2),
                    });

                REQUIRE(*ham1 == *ham2);
                REQUIRE(*ham1 != *ham3);
                REQUIRE(*ham2 != *ham3);
                REQUIRE(*ham2 != *ham4);
                REQUIRE(*ham1 != *ham5);
            }

            DYNAMIC_SECTION("getWires - "
                            << StateVectorToName<StateVectorT>::name) {
                auto Z0 = std::make_shared<NamedObsT>("PauliZ",
                                                      std::vector<size_t>{0});
                auto Z5 = std::make_shared<NamedObsT>("PauliZ",
                                                      std::vector<size_t>{5});
                auto Z9 = std::make_shared<NamedObsT>("PauliZ",
                                                      std::vector<size_t>{9});

                auto ham1 = HamiltonianT::create({0.8, 0.5, 0.7}, {Z0, Z5, Z9});

                REQUIRE(ham1->getWires() == std::vector<size_t>{0, 5, 9});
            }

            DYNAMIC_SECTION("applyInPlace must fail - "
                            << StateVectorToName<StateVectorT>::name) {

                auto ham =
                    HamiltonianT::create({PrecisionT{1.0}, h, h}, {zz, x1, x2});
                auto st_data = createZeroState<ComplexT>(2);

                StateVectorT state_vector(st_data.data(), st_data.size());

                REQUIRE_THROWS_AS(ham->applyInPlace(state_vector),
                                  LightningException);
            }
        }
        testHamiltonianBase<typename TypeList::Next>();
    }
}

TEST_CASE("Methods implemented in the HamiltonianBase class",
          "[HamiltonianBase]") {
    if constexpr (BACKEND_FOUND) {
        testHamiltonianBase<TestStateVectorBackends>();
    }
}

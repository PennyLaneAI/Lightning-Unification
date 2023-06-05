#include "ObservablesLQubit.hpp"
#include "TestHelpers.hpp"
#include "TestStateVectors.hpp" // StateVectorManagedAndPrecision, StateVectorRawAndPrecision

#include <catch2/catch.hpp>

// using namespace Pennylane;
/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit::Observables;
using Pennylane::LightningQubit::Util::StateVectorManagedAndPrecision;
using Pennylane::LightningQubit::Util::StateVectorRawAndPrecision;
using Pennylane::Util::LightningException;
} // namespace
/// @endcond

TEMPLATE_PRODUCT_TEST_CASE("NamedObs", "[Observables]",
                           (StateVectorManagedAndPrecision,
                            StateVectorRawAndPrecision),
                           (float, double)) {
    using StateVectorT = typename TestType::StateVector;
    using PrecisionT = typename TestType::Precision;
    using NamedObsT = NamedObs<StateVectorT, PrecisionT>;

    SECTION("Non-Default constructibility") {
        REQUIRE(!std::is_constructible_v<NamedObsT>);
    }

    SECTION("Constructibility") {
        REQUIRE(std::is_constructible_v<NamedObsT, std::string,
                                        std::vector<size_t>>);
    }

    SECTION("Constructibility - optional parameters") {
        REQUIRE(
            std::is_constructible_v<NamedObsT, std::string, std::vector<size_t>,
                                    std::vector<PrecisionT>>);
    }

    SECTION("Copy constructibility") {
        REQUIRE(std::is_copy_constructible_v<NamedObsT>);
    }

    SECTION("Move constructibility") {
        REQUIRE(std::is_move_constructible_v<NamedObsT>);
    }

    SECTION("NamedObs only accepts correct arguments") {
        REQUIRE_THROWS_AS(NamedObsT("PauliX", {}), LightningException);
        REQUIRE_THROWS_AS(NamedObsT("PauliX", {0, 3}), LightningException);

        REQUIRE_THROWS_AS(NamedObsT("RX", {0}), LightningException);
        REQUIRE_THROWS_AS(NamedObsT("RX", {0, 1, 2, 3}), LightningException);
        REQUIRE_THROWS_AS(
            NamedObsT("RX", {0}, std::vector<PrecisionT>{0.3, 0.4}),
            LightningException);
        REQUIRE_NOTHROW(
            NamedObsT("Rot", {0}, std::vector<PrecisionT>{0.3, 0.4, 0.5}));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("HermitianObs", "[Observables]",
                           (StateVectorManagedAndPrecision,
                            StateVectorRawAndPrecision),
                           (float, double)) {
    using StateVectorT = typename TestType::StateVector;
    using PrecisionT = typename TestType::Precision;
    using MatrixT = std::vector<std::complex<PrecisionT>>;
    using HermitianObsT = HermitianObs<StateVectorT, PrecisionT>;

    SECTION("Non-Default constructibility") {
        REQUIRE(!std::is_constructible_v<HermitianObsT>);
    }

    SECTION("Constructibility") {
        REQUIRE(std::is_constructible_v<HermitianObsT, MatrixT,
                                        std::vector<size_t>>);
    }

    SECTION("Copy constructibility") {
        REQUIRE(std::is_copy_constructible_v<HermitianObsT>);
    }

    SECTION("Move constructibility") {
        REQUIRE(std::is_move_constructible_v<HermitianObsT>);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("TensorProdObs", "[Observables]",
                           (StateVectorManagedAndPrecision,
                            StateVectorRawAndPrecision),
                           (float, double)) {
    using StateVectorT = typename TestType::StateVector;
    using PrecisionT = typename TestType::Precision;
    using TensorProdObsT = TensorProdObs<StateVectorT, PrecisionT>;
    using NamedObsT = NamedObs<StateVectorT, PrecisionT>;
    using HermitianObsT = HermitianObs<StateVectorT, PrecisionT>;

    SECTION("Constructibility - NamedObs") {
        REQUIRE(
            std::is_constructible_v<TensorProdObsT,
                                    std::vector<std::shared_ptr<NamedObsT>>>);
    }

    SECTION("Constructibility - HermitianObs") {
        REQUIRE(std::is_constructible_v<
                TensorProdObsT, std::vector<std::shared_ptr<HermitianObsT>>>);
    }

    SECTION("Copy constructibility") {
        REQUIRE(std::is_copy_constructible_v<TensorProdObsT>);
    }

    SECTION("Move constructibility") {
        REQUIRE(std::is_move_constructible_v<TensorProdObsT>);
    }
}
TEMPLATE_PRODUCT_TEST_CASE("Hamiltonian", "[Observables]",
                           (StateVectorManagedAndPrecision,
                            StateVectorRawAndPrecision),
                           (float, double)) {
    using StateVectorT = typename TestType::StateVector;
    using PrecisionT = typename TestType::Precision;
    using TensorProdObsT = TensorProdObs<StateVectorT, PrecisionT>;
    using NamedObsT = NamedObs<StateVectorT, PrecisionT>;
    using HermitianObsT = HermitianObs<StateVectorT, PrecisionT>;
    using HamiltonianT = Hamiltonian<StateVectorT, PrecisionT>;

    SECTION("Constructibility - NamedObs") {
        REQUIRE(
            std::is_constructible_v<HamiltonianT, std::vector<PrecisionT>,
                                    std::vector<std::shared_ptr<NamedObsT>>>);
    }

    SECTION("Constructibility - HermitianObs") {
        REQUIRE(std::is_constructible_v<
                HamiltonianT, std::vector<PrecisionT>,
                std::vector<std::shared_ptr<HermitianObsT>>>);
    }

    SECTION("Constructibility - TensorProdObsT") {
        REQUIRE(std::is_constructible_v<
                HamiltonianT, std::vector<PrecisionT>,
                std::vector<std::shared_ptr<TensorProdObsT>>>);
    }

    SECTION("Copy constructibility") {
        REQUIRE(std::is_copy_constructible_v<HamiltonianT>);
    }

    SECTION("Move constructibility") {
        REQUIRE(std::is_move_constructible_v<HamiltonianT>);
    }
}

TEMPLATE_TEST_CASE("Hamiltonian::ApplyInPlace<StateVectorLQubitManaged>",
                   "[Observables]", float, double) {
    using PrecisionT = TestType;
    using ComplexT = std::complex<PrecisionT>;
    using StateVectorT = StateVectorLQubitManaged<PrecisionT>;
    using TensorProdObsT = TensorProdObs<StateVectorT, PrecisionT>;
    using NamedObsT = NamedObs<StateVectorT, PrecisionT>;
    using HamiltonianT = Hamiltonian<StateVectorT, PrecisionT>;

    const auto h = PrecisionT{0.809}; // half of the golden ratio

    auto zz = std::make_shared<TensorProdObsT>(
        std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{0}),
        std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{1}));

    auto x1 = std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{0});
    auto x2 = std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{1});

    auto ham = HamiltonianT::create({PrecisionT{1.0}, h, h}, {zz, x1, x2});

    SECTION("ApplyInPlace", "[Apply Method]") {
        SECTION("Hamiltonian applies correctly to |+->") {
            auto st_data = createProductState<PrecisionT>("+-");
            StateVectorT state_vector(st_data.data(), st_data.size());

            ham->applyInPlace(state_vector);

            auto expected = std::vector<ComplexT>{
                ComplexT{0.5, 0.0},
                ComplexT{0.5, 0.0},
                ComplexT{-0.5, 0.0},
                ComplexT{-0.5, 0.0},
            };

            REQUIRE(isApproxEqual(state_vector.getData(),
                                  state_vector.getLength(), expected.data(),
                                  expected.size()));
        }

        SECTION("Hamiltonian applies correctly to |01>") {
            auto st_data = createProductState<PrecisionT>("01");
            StateVectorT state_vector(st_data.data(), st_data.size());

            ham->applyInPlace(state_vector);

            auto expected = std::vector<ComplexT>{
                ComplexT{h, 0.0},
                ComplexT{-1.0, 0.0},
                ComplexT{0.0, 0.0},
                ComplexT{h, 0.0},
            };

            REQUIRE(isApproxEqual(state_vector.getData(),
                                  state_vector.getLength(), expected.data(),
                                  expected.size()));
        }
    }
}

TEMPLATE_TEST_CASE("Hamiltonian::ApplyInPlace<StateVectorLQubitRaw>",
                   "[Observables]", float, double) {
    using PrecisionT = TestType;
    using StateVectorT = StateVectorLQubitRaw<PrecisionT>;
    using TensorProdObsT = TensorProdObs<StateVectorT, PrecisionT>;
    using NamedObsT = NamedObs<StateVectorT, PrecisionT>;
    using HamiltonianT = Hamiltonian<StateVectorT, PrecisionT>;

    const auto h = PrecisionT{0.809}; // half of the golden ratio

    auto zz = std::make_shared<TensorProdObsT>(
        std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{0}),
        std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{1}));

    auto x1 = std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{0});
    auto x2 = std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{1});

    auto ham = HamiltonianT::create({PrecisionT{1.0}, h, h}, {zz, x1, x2});

    SECTION("ApplyInPlace", "[Not implemented]") {
        auto st_data = createProductState<PrecisionT>("+-");
        StateVectorT state_vector(st_data.data(), st_data.size());

        REQUIRE_THROWS_AS(ham->applyInPlace(state_vector), LightningException);
    }
}

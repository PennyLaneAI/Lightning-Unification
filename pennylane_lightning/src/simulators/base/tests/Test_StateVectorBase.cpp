#include <complex>
#include <vector>

#include <iostream>
#include <random>

#include <catch2/catch.hpp>

#include "TestHelpers.hpp" // createZeroState, createRandomStateVectorData
#include "TypeList.hpp"

/**
 * @file
 *  Tests for functionality defined in the StateVectorBase class.
 */

/// @cond DEV
namespace {
using namespace Pennylane::Util;
} // namespace
/// @endcond

#ifdef _ENABLE_PLQUBIT
constexpr bool BACKEND_FOUND = true;

#include "TestHelpersStateVectors.hpp" // TestStateVectorBackends, StateVectorToName
#include "TestHelpersWires.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit::Util;
} // namespace
/// @endcond

#elif _ENABLE_PLKOKKOS == 1
constexpr bool BACKEND_FOUND = true;

#include "TestHelpersStateVectors.hpp" // TestStateVectorBackends, StateVectorToName
#include "TestHelpersWires.hpp"

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

template <typename TypeList> void testStateVectorBase() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        using ComplexT = StateVectorT::ComplexT;
        using VectorT = std::vector<ComplexT>;

        const size_t num_qubits = 4;
        VectorT st_data = createZeroStateComplex<ComplexT>(num_qubits);

        StateVectorT state_vector(st_data.data(), st_data.size());

        DYNAMIC_SECTION("Methods implemented in the base class - "
                        << StateVectorToName<StateVectorT>::name) {
            REQUIRE(state_vector.getNumQubits() == 4);
            REQUIRE(state_vector.getLength() == 16);
        }
        testStateVectorBase<typename TypeList::Next>();
    }
}

TEST_CASE("StateVectorBase", "[StateVectorBase]") {
    if constexpr (BACKEND_FOUND) {
        testStateVectorBase<TestStateVectorBackends>();
    }
}

template <typename TypeList> void testApplyOperations() {
    if constexpr (!std::is_same_v<TypeList, void>) {
        std::mt19937_64 re{1337};
        using StateVectorT = typename TypeList::Type;
        using PrecisionT = typename StateVectorT::PrecisionT;
        using ComplexT = typename StateVectorT::ComplexT;
        using VectorT = std::vector<ComplexT>;

        const size_t num_qubits = 3;

        DYNAMIC_SECTION("Apply operations without parameters - "
                        << StateVectorToName<StateVectorT>::name) {
            VectorT st_data_1 =
                createRandomStateVectorComplex<PrecisionT, ComplexT>(
                    re, num_qubits);
            VectorT st_data_2 = st_data_1;

            StateVectorT state_vector_1(st_data_1.data(), st_data_1.size());
            StateVectorT state_vector_2(st_data_2.data(), st_data_2.size());

            state_vector_1.applyOperations({"PauliX", "PauliY"}, {{0}, {1}},
                                           {false, false});

            state_vector_2.applyOperation("PauliX", {0}, false);
            state_vector_2.applyOperation("PauliY", {1}, false);

#if _ENABLE_PLKOKKOS == 1
            REQUIRE(isApproxEqual(
                state_vector_1.getData().data(), state_vector_1.getLength(),
                state_vector_2.getData().data(), state_vector_2.getLength()));
#else
            REQUIRE(isApproxEqual(
                state_vector_1.getData(), state_vector_1.getLength(),
                state_vector_2.getData(), state_vector_2.getLength()));
#endif
        }

        DYNAMIC_SECTION("Apply operations with parameters - "
                        << StateVectorToName<StateVectorT>::name) {
            VectorT st_data_1 =
                createRandomStateVectorComplex<PrecisionT, ComplexT>(
                    re, num_qubits);
            VectorT st_data_2 = st_data_1;

            StateVectorT state_vector_1(st_data_1.data(), st_data_1.size());
            StateVectorT state_vector_2(st_data_2.data(), st_data_2.size());

            state_vector_1.applyOperations({"RX", "RY"}, {{0}, {1}},
                                           {false, false}, {{0.1}, {0.2}});

            state_vector_2.applyOperation("RX", {0}, false, {0.1});
            state_vector_2.applyOperation("RY", {1}, false, {0.2});

#if _ENABLE_PLKOKKOS == 1
            REQUIRE(isApproxEqual(
                state_vector_1.getData().data(), state_vector_1.getLength(),
                state_vector_2.getData().data(), state_vector_2.getLength()));
#else
            REQUIRE(isApproxEqual(
                state_vector_1.getData(), state_vector_1.getLength(),
                state_vector_2.getData(), state_vector_2.getLength()));
#endif
        }
        testApplyOperations<typename TypeList::Next>();
    }
}

TEST_CASE("StateVectorBase::applyOperations", "[applyOperations]") {
    if constexpr (BACKEND_FOUND) {
        testApplyOperations<TestStateVectorBackends>();
    }
}

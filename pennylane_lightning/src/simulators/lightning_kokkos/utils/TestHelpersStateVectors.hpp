#pragma once
/**
 * @file
 * This file defines the necessary functionality to test over LQubit State
 * Vectors.
 */
#include "StateVectorKokkos.hpp"
#include "TypeList.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos;
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos::Util {

template <class StateVector> struct StateVectorToName;

template <> struct StateVectorToName<StateVectorKokkos<float>> {
    constexpr static auto name = "StateVectorKokkos<float>";
};
template <> struct StateVectorToName<StateVectorKokkos<double>> {
    constexpr static auto name = "StateVectorKokkos<double>";
};

using TestStateVectorBackends = Pennylane::Util::TypeList<
    StateVectorKokkos<float>, StateVectorKokkos<double>, void>;
} // namespace Pennylane::LightningKokkos::Util
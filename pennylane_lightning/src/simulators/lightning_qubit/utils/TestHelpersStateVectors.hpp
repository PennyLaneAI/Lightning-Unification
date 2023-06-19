#pragma once
/**
 * @file
 * This file defines the necessary functionality to test over LQubit State
 * Vectors.
 */
#include "StateVectorLQubitManaged.hpp"
#include "StateVectorLQubitRaw.hpp"
#include "TypeList.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit::Util {

template <class StateVector> struct StateVectorToName;

template <> struct StateVectorToName<StateVectorLQubitManaged<float>> {
    constexpr static auto name = "StateVectorLQubitManaged<float>";
};
template <> struct StateVectorToName<StateVectorLQubitManaged<double>> {
    constexpr static auto name = "StateVectorLQubitManaged<double>";
};
template <> struct StateVectorToName<StateVectorLQubitRaw<float>> {
    constexpr static auto name = "StateVectorLQubitRaw<float>";
};
template <> struct StateVectorToName<StateVectorLQubitRaw<double>> {
    constexpr static auto name = "StateVectorLQubitRaw<double>";
};

using TestStateVectorBackends = Pennylane::Util::TypeList<
    StateVectorLQubitManaged<float>, StateVectorLQubitManaged<double>,
    StateVectorLQubitRaw<float>, StateVectorLQubitRaw<double>, void>;
} // namespace Pennylane::LightningQubit::Util
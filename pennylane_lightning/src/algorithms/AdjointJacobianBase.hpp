// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * @file AdjointJacobianBase.hpp
 * Defines the base class to support the adjoint Jacobian differentiation
 * method.
 */
#pragma once

#include <span>

#include "JacobianData.hpp"
#include "Observables.hpp"

namespace Pennylane::Algorithms {
/**
 * @brief Adjoint Jacobian evaluator following the method of arXiV:2009.02823.
 *
 * @tparam StateVectorT State vector type.
 */
template <class StateVectorT, class Derived> class AdjointJacobianBase {
  private:
    using ComplexT = typename StateVectorT::ComplexT;
    using PrecisionT = typename StateVectorT::PrecisionT;

  protected:
    AdjointJacobianBase() = default;
    AdjointJacobianBase(const AdjointJacobianBase &) = default;
    AdjointJacobianBase(AdjointJacobianBase &&) noexcept = default;
    AdjointJacobianBase &operator=(const AdjointJacobianBase &) = default;
    AdjointJacobianBase &operator=(AdjointJacobianBase &&) noexcept = default;

    /**
     * @brief Apply all operations from given
     * `%OpsData<StateVectorT>` object to `%UpdatedStateVectorT`.
     *
     * @tparam UpdatedStateVectorT
     * @param state Statevector to be updated.
     * @param operations Operations to apply.
     * @param adj Take the adjoint of the given operations.
     */
    template <class UpdatedStateVectorT>
    inline void applyOperations(UpdatedStateVectorT &state,
                                const OpsData<StateVectorT> &operations,
                                bool adj = false) {
        for (size_t op_idx = 0; op_idx < operations.getOpsName().size();
             op_idx++) {
            state.applyOperation(operations.getOpsName()[op_idx],
                                 operations.getOpsWires()[op_idx],
                                 operations.getOpsInverses()[op_idx] ^ adj,
                                 operations.getOpsParams()[op_idx]);
        }
    }

    /**
     * @brief Apply the adjoint indexed operation from
     * `%OpsData<StateVectorT>` object to `%UpdatedStateVectorT`.
     *
     * @tparam UpdatedStateVectorT updated state vector type.
     * @param state Statevector to be updated.
     * @param operations Operations to apply.
     * @param op_idx Adjointed operation index to apply.
     */
    template <class UpdatedStateVectorT>
    inline void applyOperationAdj(UpdatedStateVectorT &state,
                                  const OpsData<StateVectorT> &operations,
                                  size_t op_idx) {
        state.applyOperation(operations.getOpsName()[op_idx],
                             operations.getOpsWires()[op_idx],
                             !operations.getOpsInverses()[op_idx],
                             operations.getOpsParams()[op_idx]);
    }

    /**
     * @brief Apply a given `%Observable<StateVectorT>` object to
     * `%StateVectorT`
     *
     * @param state Statevector to be updated.
     * @param observable Observable to apply.
     */
    inline void applyObservable(StateVectorT &state,
                                const Observable<StateVectorT> &observable) {
        observable.applyInPlace(state);
    }

    /**
     * @brief Calculates the statevector's Jacobian for the selected set
     * of parametric gates.
     *
     * @param jac Preallocated vector for Jacobian data results.
     * @param jd JacobianData represents the QuantumTape to differentiate.
     * @param apply_operations Indicate whether to apply operations to tape.psi
     * prior to calculation.
     */
    inline void adjointJacobian(std::span<PrecisionT> jac,
                                const JacobianData<StateVectorT> &jd,
                                bool apply_operations = false) {
        return static_cast<Derived *>(this)->adjointJacobian(jac, jd,
                                                             apply_operations);
    }
};
} // namespace Pennylane::Algorithms
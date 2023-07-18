#pragma once
#include "AdjointJacobianBase.hpp"
#include "ObservablesKokkos.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos::Observables;
using namespace Pennylane::Algorithms;
using Pennylane::LightningKokkos::Util::getImagOfComplexInnerProduct;
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos::Algorithms {

/**
 * @brief Kokkos-enabled adjoint Jacobian evaluator following the method of
 * arXiV:2009.02823
 *
 * @tparam StateVectorT State vector type.
 */
template <class StateVectorT>
class AdjointJacobian final
    : public AdjointJacobianBase<StateVectorT, AdjointJacobian<StateVectorT>> {
  private:
    using ComplexT = typename StateVectorT::ComplexT;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using BaseType =
        AdjointJacobianBase<StateVectorT, AdjointJacobian<StateVectorT>>;

    /**
     * @brief Utility method to update the Jacobian at a given index by
     * calculating the overlap between two given states.
     *
     * @param sv1 Statevector <sv1|. Data will be conjugated.
     * @param sv2 Statevector |sv2>
     * @param jac Jacobian receiving the values.
     * @param scaling_coeff Generator coefficient for given gate derivative.
     * @param obs_index Observable index position of Jacobian to update.
     * @param param_index Parameter index position of Jacobian to update.
     */
    inline void updateJacobian(StateVectorT &sv1, StateVectorT &sv2,
                               std::vector<std::vector<PrecisionT>> &jac,
                               PrecisionT scaling_coeff, size_t obs_index,
                               size_t param_index) {
        jac[obs_index][param_index] = -2 * scaling_coeff *
                                      getImagOfComplexInnerProduct<PrecisionT>(
                                          sv1.getData(), sv2.getData());
    }

    /**
     * @brief Utility method to apply all operations from given
     * `%OpsData<StateVectorT>` object to
     * `%StateVectorT`
     *
     * @param state Statevector to be updated.
     * @param operations Operations to apply.
     * @param adj Take the adjoint of the given operations.
     */
    inline void applyOperations(StateVectorT &state,
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
     * @brief Utility method to apply the adjoint indexed operation from
     * `%OpsData<StateVectorT>` object to
     * `%StateVectorT`.
     *
     * @param state Statevector to be updated.
     * @param operations Operations to apply.
     * @param op_idx Adjointed operation index to apply.
     */
    inline void applyOperationAdj(StateVectorT &state,
                                  const OpsData<StateVectorT> &operations,
                                  size_t op_idx) {
        state.applyOperation(operations.getOpsName()[op_idx],
                             operations.getOpsWires()[op_idx],
                             !operations.getOpsInverses()[op_idx],
                             operations.getOpsParams()[op_idx]);
    }

    /**
     * @brief Utility method to apply a given operations from given
     * `%ObsDatum<PrecisionT>` object
     * to
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
     * @brief Application of observables to given
     * statevectors
     *
     * @param states Vector of statevector copies, one per observable.
     * @param reference_state Reference statevector
     * @param observables Vector of observables to apply to each statevector.
     */
    inline void applyObservables(
        std::vector<StateVectorT> &states, const StateVectorT &reference_state,
        const std::vector<std::shared_ptr<Observable<StateVectorT>>>
            &observables) {
        // clang-format off
        // Globally scoped exception value to be captured within OpenMP block.
        // See the following for OpenMP design decisions:
        // https://www.openmp.org/wp-content/uploads/openmp-examples-4.5.0.pdf
        std::exception_ptr ex = nullptr;
        size_t num_observables = observables.size();
            for (size_t h_i = 0; h_i < num_observables; h_i++) {
                try {
                    states[h_i].updateData(reference_state);
                    applyObservable(states[h_i], *observables[h_i]);
                } catch (...) {
                    ex = std::current_exception();
                }
            }
        if (ex) {
            std::rethrow_exception(ex);
        }
        // clang-format on
    }

    /**
     * @brief Application of adjoint operations to
     * statevectors.
     *
     * @param states Vector of all statevectors; 1 per observable
     * @param operations Operations list.
     * @param op_idx Index of given operation within operations list to take
     * adjoint of.
     */
    inline void applyOperationsAdj(std::vector<StateVectorT> &states,
                                   const OpsData<StateVectorT> &operations,
                                   size_t op_idx) {
        // clang-format off
        // Globally scoped exception value to be captured within OpenMP block.
        // See the following for OpenMP design decisions:
        // https://www.openmp.org/wp-content/uploads/openmp-examples-4.5.0.pdf
        std::exception_ptr ex = nullptr;
        size_t num_states = states.size();
            for (size_t obs_idx = 0; obs_idx < num_states; obs_idx++) {
                try {
                    applyOperationAdj(states[obs_idx], operations, op_idx);
                } catch (...) {
                    ex = std::current_exception();
                }
            }
        if (ex) {
            std::rethrow_exception(ex);
        }
        // clang-format on
    }

    /**
     * @brief Inline utility to assist with getting the Jacobian index offset.
     *
     * @param obs_index
     * @param tp_index
     * @param tp_size
     * @return size_t
     */
    inline auto getJacIndex(size_t obs_index, size_t tp_index, size_t tp_size)
        -> size_t {
        return obs_index * tp_size + tp_index;
    }

    /**
     * @brief Applies the gate generator for a given parameteric gate. Returns
     * the associated scaling coefficient.
     *
     * @param sv Statevector data to operate upon.
     * @param op_name Name of parametric gate.
     * @param wires Wires to operate upon.
     * @param adj Indicate whether to take the adjoint of the operation.
     * @return PrecisionT Generator scaling coefficient.
     */
    inline auto applyGenerator(StateVectorT &sv, const std::string &op_name,
                               const std::vector<size_t> &wires, const bool adj)
        -> PrecisionT {
        return sv.applyGenerator(op_name, wires, adj);
    }

  public:
    AdjointJacobian() = default;

    /**
     * @brief Utility to create a given operations object.
     *
     * @param ops_name Name of operations.
     * @param ops_params Parameters for each operation in ops_name.
     * @param ops_wires Wires for each operation in ops_name.
     * @param ops_inverses Indicate whether to take adjoint of each operation in
     * ops_name.
     * @param ops_matrices Matrix definition of an operation if unsupported.
     * @return const
     * OpsData<StateVectorT>
     */
    auto
    createOpsData(const std::vector<std::string> &ops_name,
                  const std::vector<std::vector<PrecisionT>> &ops_params,
                  const std::vector<std::vector<size_t>> &ops_wires,
                  const std::vector<bool> &ops_inverses,
                  const std::vector<std::vector<ComplexT>> &ops_matrices = {{}})
        -> OpsData<StateVectorT> {
        return {ops_name, ops_params, ops_wires, ops_inverses, ops_matrices};
    }

    /**
     * @brief Calculates the Jacobian for the statevector for the selected set
     * of parametric gates.
     *
     * For the statevector data associated with `psi` of length `num_elements`,
     * we make internal copies, one per required observable. The `operations`
     * will be applied to the internal statevector copies, with the operation
     * indices participating in the gradient calculations given in
     * `trainableParams`, and the overall number of parameters for the gradient
     * calculation provided within `num_params`. The resulting row-major ordered
     * `jac` matrix representation will be of size `jd.getSizeStateVec() *
     * jd.getObservables().size()`. OpenMP is used to enable independent
     * operations to be offloaded to threads.
     *
     * @param jac Preallocated vector for Jacobian data results.
     * @param jd JacobianData represents the QuantumTape to differentiate.
     * @param apply_operations Indicate whether to apply operations to tape.psi
     * prior to calculation.
     */
    void adjointJacobian(std::span<PrecisionT> jac,
                         const JacobianData<StateVectorT> &jd,
                         bool apply_operations = false) {
        const OpsData<StateVectorT> &ops = jd.getOperations();

        const auto &obs = jd.getObservables();
        const size_t num_observables = obs.size();

        // We can assume the trainable params are sorted (from Python)
        const std::vector<size_t> &tp = jd.getTrainableParams();
        const size_t tp_size = tp.size();

        if (!jd.hasTrainableParams()) {
            return;
        }

        StateVectorKokkos<PrecisionT> ref_data(jd.getPtrStateVec(),
            jd.getSizeStateVec());

        PL_ABORT_IF_NOT(
            jac.size() == tp_size * num_observables,
            "The size of preallocated jacobian must be same as "
            "the number of trainable parameters times the number of "
            "observables provided.");

        std::vector<std::vector<PrecisionT>> jac_data(
            num_observables, std::vector<PrecisionT>(tp_size, 0.0));

        adjointJacobian(ref_data, jac_data, obs, ops, tp, apply_operations);

        for (size_t obs_idx = 0; obs_idx < num_observables; obs_idx++) {
            const size_t mat_row_idx = obs_idx * tp_size;
            for (size_t op_idx = 0; op_idx < tp_size; op_idx++) {
                jac[mat_row_idx + op_idx] = jac_data[obs_idx][op_idx];
            }
        }
    }
    /**
     * @brief Calculates the Jacobian for the statevector for the selected set
     * of parametric gates.
     *
     * For the statevector data associated with `psi` of length `num_elements`,
     * we make internal copies to a `%StateVectorT` object, with
     * one per required observable. The `operations` will be applied to the
     * internal statevector copies, with the operation indices participating in
     * the gradient calculations given in `trainableParams`, and the overall
     * number of parameters for the gradient calculation provided within
     * `num_params`. The resulting row-major ordered `jac` matrix representation
     * will be of size `trainableParams.size() * observables.size()`.
     *
     * @param ref_data Pointer to the statevector data.
     * @param length Length of the statevector data.
     * @param jac Preallocated vector for Jacobian data results.
     * @param obs ObservableKokkos for which to calculate Jacobian.
     * @param ops Operations used to create given state.
     * @param trainableParams List of parameters participating in Jacobian
     * calculation.
     * @param apply_operations Indicate whether to apply operations to psi prior
     * to calculation.
     */
    void adjointJacobian(
        const StateVectorT &ref_data, std::vector<std::vector<PrecisionT>> &jac,
        const std::vector<std::shared_ptr<Observable<StateVectorT>>> &obs,
        const OpsData<StateVectorT> &ops,
        const std::vector<size_t> &trainableParams,
        bool apply_operations = false) {
        PL_ABORT_IF(trainableParams.empty(),
                    "No trainable parameters provided.");

        const std::vector<std::string> &ops_name = ops.getOpsName();
        const size_t num_observables = obs.size();

        const size_t tp_size = trainableParams.size();
        const size_t num_param_ops = ops.getNumParOps();

        // Track positions within par and non-par operations
        size_t trainableParamNumber = tp_size - 1;
        size_t current_param_idx =
            num_param_ops - 1; // total number of parametric ops
        auto tp_it = trainableParams.rbegin();
        const auto tp_rend = trainableParams.rend();

        // Create $U_{1:p}\vert \lambda \rangle$
        StateVectorT lambda(ref_data.getNumQubits());
        lambda.DeviceToDevice(ref_data.getData());

        // Apply given operations to statevector if requested
        if (apply_operations) {
            applyOperations(lambda, ops);
        }

        // Create observable-applied state-vectors
        std::vector<StateVectorT> H_lambda(num_observables,
                                           StateVectorT(lambda.getNumQubits()));
        applyObservables(H_lambda, lambda, obs);

        StateVectorT mu(lambda.getNumQubits());

        for (int op_idx = static_cast<int>(ops_name.size() - 1); op_idx >= 0;
             op_idx--) {
            PL_ABORT_IF(ops.getOpsParams()[op_idx].size() > 1,
                        "The operation is not supported using the adjoint "
                        "differentiation method");
            if ((ops_name[op_idx] == "QubitStateVector") ||
                (ops_name[op_idx] == "BasisState")) {
                continue;
            }
            if (tp_it == tp_rend) {
                break; // All done
            }
            mu.updateData(lambda);
            applyOperationAdj(lambda, ops, op_idx);

            if (ops.hasParams(op_idx)) {
                if (current_param_idx == *tp_it) {
                    const PrecisionT scalingFactor =
                        applyGenerator(mu, ops.getOpsName()[op_idx],
                                       ops.getOpsWires()[op_idx],
                                       !ops.getOpsInverses()[op_idx]) *
                        (ops.getOpsInverses()[op_idx] ? -1 : 1);

                    for (size_t obs_idx = 0; obs_idx < num_observables;
                         obs_idx++) {
                        updateJacobian(H_lambda[obs_idx], mu, jac,
                                       scalingFactor, obs_idx,
                                       trainableParamNumber);
                    }
                    trainableParamNumber--;
                    ++tp_it;
                }
                current_param_idx--;
            }
            applyOperationsAdj(H_lambda, ops, static_cast<size_t>(op_idx));
        }
    }
};

} // namespace Pennylane::LightningKokkos::Algorithms

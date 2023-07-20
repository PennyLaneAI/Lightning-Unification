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
                                          sv1.getView(), sv2.getView());
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
        lambda.DeviceToDevice(ref_data.getView());

        // Apply given operations to statevector if requested
        if (apply_operations) {
            this->applyOperations(lambda, ops);
        }

        // Create observable-applied state-vectors
        std::vector<StateVectorT> H_lambda(num_observables,
                                           StateVectorT(lambda.getNumQubits()));
        this->applyObservables(H_lambda, lambda, obs);

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
            this->applyOperationAdj(lambda, ops, op_idx);

            if (ops.hasParams(op_idx)) {
                if (current_param_idx == *tp_it) {
                    const PrecisionT scalingFactor =
                        this->applyGenerator(mu, ops.getOpsName()[op_idx],
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
            this->applyOperationsAdj(H_lambda, ops,
                                     static_cast<size_t>(op_idx));
        }
    }
};

} // namespace Pennylane::LightningKokkos::Algorithms

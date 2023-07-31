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
#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "ExpValFunctors.hpp"
#include "LinearAlgebraKokkos.hpp" // getRealOfComplexInnerProduct
#include "MeasurementsBase.hpp"
#include "MeasuresFunctors.hpp"
#include "Observables.hpp"
#include "StateVectorKokkos.hpp"
#include "Util.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Measures;
using namespace Pennylane::Observables;
using Pennylane::LightningKokkos::StateVectorKokkos;
using Pennylane::LightningKokkos::Util::getRealOfComplexInnerProduct;
using Pennylane::Util::exp2;
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos::Measures {

template <class StateVectorT>
class Measurements final
    : public MeasurementsBase<StateVectorT, Measurements<StateVectorT>> {

  private:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using BaseType = MeasurementsBase<StateVectorT, Measurements<StateVectorT>>;

    using KokkosExecSpace = typename StateVectorT::KokkosExecSpace;
    using KokkosVector = typename StateVectorT::KokkosVector;
    using KokkosSizeTVector = typename StateVectorT::KokkosSizeTVector;
    using UnmanagedSizeTHostView =
        typename StateVectorT::UnmanagedSizeTHostView;
    using UnmanagedConstComplexHostView =
        typename StateVectorT::UnmanagedConstComplexHostView;
    using UnmanagedConstSizeTHostView =
        typename StateVectorT::UnmanagedConstSizeTHostView;
    using UnmanagedPrecisionHostView =
        typename StateVectorT::UnmanagedPrecisionHostView;

    using ExpValFunc = std::function<PrecisionT(
        const std::vector<size_t> &, const std::vector<PrecisionT> &)>;
    using ExpValMap = std::unordered_map<std::string, ExpValFunc>;

    ExpValMap expval_funcs;

  public:
    explicit Measurements(const StateVectorT &statevector)
        : BaseType{statevector},
          expval_funcs{{"Identity",
                        [&](auto &&wires, auto &&params) {
                            return getExpectationValueIdentity(
                                std::forward<decltype(wires)>(wires),
                                std::forward<decltype(params)>(params));
                        }},
                       {"PauliX",
                        [&](auto &&wires, auto &&params) {
                            return getExpectationValuePauliX(
                                std::forward<decltype(wires)>(wires),
                                std::forward<decltype(params)>(params));
                        }},
                       {"PauliY",
                        [&](auto &&wires, auto &&params) {
                            return getExpectationValuePauliY(
                                std::forward<decltype(wires)>(wires),
                                std::forward<decltype(params)>(params));
                        }},
                       {"PauliZ",
                        [&](auto &&wires, auto &&params) {
                            return getExpectationValuePauliZ(
                                std::forward<decltype(wires)>(wires),
                                std::forward<decltype(params)>(params));
                        }},
                       {"Hadamard", [&](auto &&wires, auto &&params) {
                            return getExpectationValueHadamard(
                                std::forward<decltype(wires)>(wires),
                                std::forward<decltype(params)>(params));
                        }}} {};

    /**
     * @brief Calculate the expectation value of a named observable.
     *
     * @param obsName observable name
     * @param wires wires the observable acts on
     * @param params parameters for the observable
     * @param gate_matrix optional matrix
     */
    PrecisionT getExpectationValue(
        const std::string &obsName, const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<PrecisionT> &params = {0.0},
        const std::vector<ComplexT> &gate_matrix = {}) {
        auto &&par = (params.empty()) ? std::vector<PrecisionT>{0.0} : params;
        auto &&local_wires =
            (gate_matrix.empty())
                ? wires
                : std::vector<size_t>{
                      wires.rbegin(),
                      wires.rend()}; // ensure wire indexing correctly preserved
                                     // for tensor-observables

        if (expval_funcs.find(obsName) != expval_funcs.end()) {
            return expval_funcs.at(obsName)(local_wires, par);
        }

        KokkosVector matrix("gate_matrix", gate_matrix.size());
        Kokkos::deep_copy(matrix, UnmanagedConstComplexHostView(
                                      gate_matrix.data(), gate_matrix.size()));
        return getExpectationValueMultiQubitOp(matrix, wires, par);
    }

    /**
     * @brief Calculate the expectation value of a matrix. Typically,
     * this function will be used for dense Hamiltonians.
     *
     * @param wires wires the observable acts on
     * @param gate_matrix optional matrix
     */
    PrecisionT getExpectationValue(const std::vector<size_t> &wires,
                                   const std::vector<ComplexT> &gate_matrix) {

        auto &&par = std::vector<PrecisionT>{0.0};
        KokkosVector matrix("gate_matrix", gate_matrix.size());
        Kokkos::deep_copy(matrix, UnmanagedConstComplexHostView(
                                      gate_matrix.data(), gate_matrix.size()));
        return getExpectationValueMultiQubitOp(matrix, wires, par);
    }

    /**
     * @brief Calculate the expectation value of a sparse Hamiltonian in CSR
     * format. Typically, this function will be used for dense hamiltonians.
     *
     * @tparam index_type integer type used as indices of the sparse matrix.
     * @param row_map   row_map array pointer. The j element encodes the
     * number of non-zeros above row j.
     * @param row_map_size  row_map array size.
     * @param indices   pointer to an array with column indices of the non-zero
     * elements.
     * @param data    pointer to an array with the non-zero elements.
     * @param data_size        number of non-zero elements.
     */
    template <class index_type>
    PrecisionT getExpectationValue(const index_type *row_map,
                                   index_type row_map_size,
                                   const index_type *indices,
                                   const ComplexT *data, index_type data_size) {
        const Kokkos::View<ComplexT *> arr_data = this->_statevector.getView();
        PrecisionT expval = 0.0;
        KokkosSizeTVector kok_row_map("row_map", row_map_size);
        KokkosSizeTVector kok_indices("indices", data_size);
        KokkosVector kok_data("data", data_size);

        Kokkos::deep_copy(kok_data,
                          UnmanagedConstComplexHostView(data, data_size));
        Kokkos::deep_copy(kok_indices,
                          UnmanagedConstSizeTHostView(indices, data_size));
        Kokkos::deep_copy(kok_row_map,
                          UnmanagedConstSizeTHostView(row_map, row_map_size));

        Kokkos::parallel_reduce(
            row_map_size - 1,
            getExpectationValueSparseFunctor<PrecisionT>(
                arr_data, kok_data, kok_indices, kok_row_map),
            expval);
        return expval;
    }

    /**
     * @brief Calculate the expectation value of a sparse Hamiltonian in CSR
     * format. Typically, this function will be used for dense hamiltonians.
     *
     * @tparam index_type integer type used as indices of the sparse matrix.
     * @param data    array with the non-zero elements.
     * @param indices   array with column indices of the non-zero
     * elements.
     * @param row_map   row_map array. The j element encodes the
     * number of non-zeros above row j.
     */
    template <class index_type>
    PrecisionT getExpectationValue(const std::vector<ComplexT> &data,
                                   const std::vector<index_type> &indices,
                                   const std::vector<index_type> &row_map) {
        return getExpectationValue(row_map.data(), row_map.size(),
                                   indices.data(), data.data(), data.size());
    }

    /**
     * @brief Calculate expectation value with respect to identity observable on
     * specified wire. For normalised states this function will always return 1.
     *
     * @param wires Wire to apply observable to.
     * @param params Not used.
     * @return Squared norm of state.
     */
    auto getExpectationValueIdentity(
        const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<PrecisionT> &params = {0.0}) {
        const size_t num_qubits = this->_statevector.getNumQubits();
        const Kokkos::View<ComplexT *> arr_data = this->_statevector.getView();
        PrecisionT expval = 0;
        Kokkos::parallel_reduce(
            exp2(num_qubits),
            getExpectationValueIdentityFunctor(arr_data, num_qubits, wires),
            expval);
        return expval;
    }

    /**
     * @brief Calculate expectation value with respect to Pauli X observable on
     * specified wire.
     *
     * @param wires Wire to apply observable to.
     * @param params Not used.
     * @return Expectation value with respect to Pauli X applied to specified
     * wire.
     */
    auto getExpectationValuePauliX(
        const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<PrecisionT> &params = {0.0}) {
        const size_t num_qubits = this->_statevector.getNumQubits();
        const Kokkos::View<ComplexT *> arr_data = this->_statevector.getView();
        PrecisionT expval = 0;
        Kokkos::parallel_reduce(
            exp2(num_qubits - 1),
            getExpectationValuePauliXFunctor(arr_data, num_qubits, wires),
            expval);
        return expval;
    }

    /**
     * @brief Calculate expectation value with respect to Pauli Y observable on
     * specified wire.
     *
     * @param wires Wire to apply observable to.
     * @param params Not used.
     * @return Expectation value with respect to Pauli Y applied to specified
     * wire.
     */
    auto getExpectationValuePauliY(
        const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<PrecisionT> &params = {0.0}) {
        const size_t num_qubits = this->_statevector.getNumQubits();
        const Kokkos::View<ComplexT *> arr_data = this->_statevector.getView();
        PrecisionT expval = 0;
        Kokkos::parallel_reduce(
            exp2(num_qubits - 1),
            getExpectationValuePauliYFunctor(arr_data, num_qubits, wires),
            expval);
        return expval;
    }

    /**
     * @brief Calculate expectation value with respect to Pauli Z observable on
     * specified wire.
     *
     * @param wires Wire to apply observable to.
     * @param params Not used.
     * @return Expectation value with respect to Pauli Z applied to specified
     * wire.
     */
    auto getExpectationValuePauliZ(
        const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<PrecisionT> &params = {0.0}) {
        const size_t num_qubits = this->_statevector.getNumQubits();
        const Kokkos::View<ComplexT *> arr_data = this->_statevector.getView();
        PrecisionT expval = 0;
        Kokkos::parallel_reduce(
            exp2(num_qubits - 1),
            getExpectationValuePauliZFunctor(arr_data, num_qubits, wires),
            expval);
        return expval;
    }

    /**
     * @brief Calculate expectation value with respect to Hadamard observable on
     * specified wire.
     *
     * @param wires Wire to apply observable to.
     * @param params Not used.
     * @return Expectation value with respect to Hadamard applied to specified
     * wire.
     */
    auto getExpectationValueHadamard(
        const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<PrecisionT> &params = {0.0}) {
        const size_t num_qubits = this->_statevector.getNumQubits();
        const Kokkos::View<ComplexT *> arr_data = this->_statevector.getView();
        PrecisionT expval = 0;
        Kokkos::parallel_reduce(
            exp2(num_qubits - 1),
            getExpectationValueHadamardFunctor(arr_data, num_qubits, wires),
            expval);
        return expval;
    }

    /**
     * @brief Calculate expectation value with respect to single qubit
     * observable on specified wire.
     *
     * @param matrix Hermitian matrix representing observable to be used.
     * @param wires Wire to apply observable to.
     * @param params Not used.
     * @return Expectation value with respect to observable applied to specified
     * wire.
     */
    auto getExpectationValueSingleQubitOp(
        const KokkosVector &matrix, const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<PrecisionT> &params = {0.0}) {
        const size_t num_qubits = this->_statevector.getNumQubits();
        Kokkos::View<ComplexT *> arr_data = this->_statevector.getView();
        PrecisionT expval = 0;
        Kokkos::parallel_reduce(
            exp2(num_qubits - 1),
            getExpectationValueSingleQubitOpFunctor<PrecisionT>(
                arr_data, num_qubits, matrix, wires),
            expval);
        return expval;
    }

    /**
     * @brief Calculate expectation value with respect to two qubit observable
     * on specified wires.
     *
     * @param matrix Hermitian matrix representing observable to be used.
     * @param wires Wires to apply observable to.
     * @param params Not used.
     * @return Expectation value with respect to observable applied to specified
     * wires.
     */
    auto getExpectationValueTwoQubitOp(
        const KokkosVector &matrix, const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<PrecisionT> &params = {0.0}) {
        const size_t num_qubits = this->_statevector.getNumQubits();
        Kokkos::View<ComplexT *> arr_data = this->_statevector.getView();
        PrecisionT expval = 0;
        Kokkos::parallel_reduce(
            exp2(num_qubits - 2),
            getExpectationValueTwoQubitOpFunctor<PrecisionT>(
                arr_data, num_qubits, matrix, wires),
            expval);
        return expval;
    }

    /**
     * @brief Calculate expectation value with respect to multi qubit observable
     * on specified wires.
     *
     * @param matrix Hermitian matrix representing observable to be used.
     * @param wires Wires to apply observable to.
     * @param params Not used.
     * @return Expectation value with respect to observable applied to specified
     * wires.
     */
    auto getExpectationValueMultiQubitOp(
        const KokkosVector &matrix, const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<PrecisionT> &params = {0.0}) {
        if (wires.size() == 1) {
            return getExpectationValueSingleQubitOp(matrix, wires, params);
        } else if (wires.size() == 2) {
            return getExpectationValueTwoQubitOp(matrix, wires, params);
        } else {
            Kokkos::View<const size_t *, Kokkos::HostSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                wires_host(wires.data(), wires.size());
            const size_t num_qubits = this->_statevector.getNumQubits();
            const Kokkos::View<ComplexT *> arr_data =
                this->_statevector.getView();

            Kokkos::View<size_t *> wires_view("wires_view", wires.size());
            Kokkos::deep_copy(wires_view, wires_host);
            PrecisionT expval = 0;
            Kokkos::parallel_reduce(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, exp2(num_qubits - wires.size())),
                getExpectationValueMultiQubitOpFunctor(arr_data, num_qubits,
                                                       matrix, wires_view),
                expval);
            return expval;
        }
    }

    /**
     * @brief Calculate expectation value for a general Observable.
     *
     * @param ob Observable.
     * @return Expectation value with respect to the given observable.
     */
    auto expval(const Observable<StateVectorT> &ob) {
        StateVectorT ob_sv(this->_statevector.getNumQubits());
        ob_sv.DeviceToDevice(this->_statevector.getView());
        ob.applyInPlace(ob_sv);
        return getRealOfComplexInnerProduct(this->_statevector.getView(),
                                            ob_sv.getView());
    }

    /**
     * @brief Expected value of an observable.
     *
     * @param matrix Square matrix in row-major order.
     * @param wires Wires where to apply the operator.
     * @return Floating point expected value of the observable.
     */
    PrecisionT expval(const std::vector<ComplexT> &matrix,
                      const std::vector<size_t> &wires) {
        StateVectorT ob_sv(this->_statevector.getNumQubits());
        ob_sv.DeviceToDevice(this->_statevector.getView());
        ob_sv.applyMatrix(matrix, wires);
        return getRealOfComplexInnerProduct(this->_statevector.getView(),
                                            ob_sv.getView());
    };

    /**
     * @brief Expected value of an observable.
     *
     * @param operation String with the operator name.
     * @param wires Wires where to apply the operator.
     * @return Floating point expected value of the observable.
     */
    PrecisionT expval(const std::string &operation,
                      const std::vector<size_t> &wires) {
        StateVectorT ob_sv(this->_statevector.getNumQubits());
        ob_sv.DeviceToDevice(this->_statevector.getView());
        ob_sv.applyOperation(operation, wires);
        return getRealOfComplexInnerProduct(this->_statevector.getView(),
                                            ob_sv.getView());
    };

    /**
     * @brief Expected value for a list of observables.
     *
     * @tparam op_type Operation type.
     * @param operations_list List of operations to measure.
     * @param wires_list List of wires where to apply the operators.
     * @return Floating point std::vector with expected values for the
     * observables.
     */
    template <typename op_type>
    std::vector<PrecisionT>
    expval(const std::vector<op_type> &operations_list,
           const std::vector<std::vector<size_t>> &wires_list) {
        PL_ABORT_IF(
            (operations_list.size() != wires_list.size()),
            "The lengths of the list of operations and wires do not match.");
        std::vector<PrecisionT> expected_value_list;

        for (size_t index = 0; index < operations_list.size(); index++) {
            expected_value_list.emplace_back(
                expval(operations_list[index], wires_list[index]));
        }

        return expected_value_list;
    }

    /**
     * @brief Expected value of a Sparse Hamiltonian.
     *
     * @tparam index_type integer type used as indices of the sparse matrix.
     * @param row_map_ptr   row_map array pointer.
     *                      The j element encodes the number of non-zeros
     above
     * row j.
     * @param row_map_size  row_map array size.
     * @param entries_ptr   pointer to an array with column indices of the
     * non-zero elements.
     * @param values_ptr    pointer to an array with the non-zero elements.
     * @param numNNZ        number of non-zero elements.
     * @return Floating point expected value of the observable.
     */
    template <class index_type>
    PrecisionT expval(const index_type *row_map_ptr,
                      const index_type row_map_size,
                      const index_type *entries_ptr, const ComplexT *values_ptr,
                      const index_type numNNZ) {
        return getExpectationValue(row_map_ptr, row_map_size, entries_ptr,
                                   values_ptr, numNNZ);
    };

    /**
     * @brief Calculate variance of a general Observable.
     *
     * @param ob Observable.
     * @return Variance with respect to the given observable.
     */
    auto var(const Observable<StateVectorT> &ob) -> PrecisionT {
        StateVectorT ob_sv(this->_statevector.getNumQubits());
        ob_sv.DeviceToDevice(this->_statevector.getView());
        ob.applyInPlace(ob_sv);

        const PrecisionT mean_square =
            getRealOfComplexInnerProduct(ob_sv.getView(), ob_sv.getView());
        const PrecisionT squared_mean = static_cast<PrecisionT>(
            std::pow(getRealOfComplexInnerProduct(this->_statevector.getView(),
                                                  ob_sv.getView()),
                     2));
        return (mean_square - squared_mean);
    }

    /**
     * @brief Variance of an observable.
     *
     * @param operation String with the operator name.
     * @param wires Wires where to apply the operator.
     * @return Floating point with the variance of the observable.
     */
    PrecisionT var(const std::string &operation,
                   const std::vector<size_t> &wires) {
        StateVectorT ob_sv(this->_statevector.getNumQubits());
        ob_sv.DeviceToDevice(this->_statevector.getView());
        ob_sv.applyOperation(operation, wires);

        const PrecisionT mean_square =
            getRealOfComplexInnerProduct(ob_sv.getView(), ob_sv.getView());
        const PrecisionT squared_mean = static_cast<PrecisionT>(
            std::pow(getRealOfComplexInnerProduct(this->_statevector.getView(),
                                                  ob_sv.getView()),
                     2));
        return (mean_square - squared_mean);
    };

    /**
     * @brief Variance of an observable.
     *
     * @param matrix Square matrix in row-major order.
     * @param wires Wires where to apply the operator.
     * @return Floating point with the variance of the observable.
     */
    PrecisionT var(const std::vector<ComplexT> &matrix,
                   const std::vector<size_t> &wires) {
        StateVectorT ob_sv(this->_statevector.getNumQubits());
        ob_sv.DeviceToDevice(this->_statevector.getView());
        ob_sv.applyMatrix(matrix, wires);

        const PrecisionT mean_square =
            getRealOfComplexInnerProduct(ob_sv.getView(), ob_sv.getView());
        const PrecisionT squared_mean = static_cast<PrecisionT>(
            std::pow(getRealOfComplexInnerProduct(this->_statevector.getView(),
                                                  ob_sv.getView()),
                     2));
        return (mean_square - squared_mean);
    };

    /**
     * @brief Variance for a list of observables.
     *
     * @tparam op_type Operation type.
     * @param operations_list List of operations to measure.
     * Square matrix in row-major order or string with the operator name.
     * @param wires_list List of wires where to apply the operators.
     * @return Floating point std::vector with the variance of the
     observables.
     */
    template <typename op_type>
    std::vector<PrecisionT>
    var(const std::vector<op_type> &operations_list,
        const std::vector<std::vector<size_t>> &wires_list) {
        PL_ABORT_IF(
            (operations_list.size() != wires_list.size()),
            "The lengths of the list of operations and wires do not match.");

        std::vector<PrecisionT> expected_value_list;

        for (size_t index = 0; index < operations_list.size(); index++) {
            expected_value_list.emplace_back(
                var(operations_list[index], wires_list[index]));
        }

        return expected_value_list;
    };

    /**
     * @brief Probabilities of each computational basis state.
     *
     * @return Floating point std::vector with probabilities
     * in lexicographic order.
     */
    auto probs() -> std::vector<PrecisionT> {
        const size_t N = this->_statevector.getLength();

        Kokkos::View<ComplexT *> arr_data = this->_statevector.getView();
        Kokkos::View<PrecisionT *> d_probability("d_probability", N);

        // Compute probability distribution from StateVector
        Kokkos::parallel_for(
            Kokkos::RangePolicy<KokkosExecSpace>(0, N),
            getProbFunctor<PrecisionT>(arr_data, d_probability));

        std::vector<PrecisionT> probabilities(N, 0);

        Kokkos::deep_copy(UnmanagedPrecisionHostView(probabilities.data(),
                                                     probabilities.size()),
                          d_probability);
        return probabilities;
    }

    /**
     * @brief Probabilities for a subset of the full system.
     *
     * @param wires Wires will restrict probabilities to a subset
     * of the full system.
     * @return Floating point std::vector with probabilities.
     * The basis columns are rearranged according to wires.
     */
    auto probs(const std::vector<size_t> &wires) {
        using MDPolicyType_2D =
            Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>;

        //  Determining probabilities for the sorted wires.
        const Kokkos::View<ComplexT *> arr_data = this->_statevector.getView();
        const size_t num_qubits = this->_statevector.getNumQubits();

        std::vector<size_t> sorted_ind_wires(wires);
        const bool is_sorted_wires =
            std::is_sorted(sorted_ind_wires.begin(), sorted_ind_wires.end());
        std::vector<size_t> sorted_wires(wires);

        if (!is_sorted_wires) {
            sorted_ind_wires = Pennylane::Util::sorting_indices(wires);
            for (size_t pos = 0; pos < wires.size(); pos++)
                sorted_wires[pos] = wires[sorted_ind_wires[pos]];
        }

        std::vector<size_t> all_indices =
            Pennylane::Util::generateBitsPatterns(sorted_wires, num_qubits);

        std::vector<size_t> all_offsets = Pennylane::Util::generateBitsPatterns(
            Pennylane::Util::getIndicesAfterExclusion(sorted_wires, num_qubits),
            num_qubits);

        Kokkos::View<PrecisionT *> d_probabilities("d_probabilities",
                                                   all_indices.size());

        Kokkos::View<size_t *> d_sorted_ind_wires("d_sorted_ind_wires",
                                                  sorted_ind_wires.size());
        Kokkos::View<size_t *> d_all_indices("d_all_indices",
                                             all_indices.size());
        Kokkos::View<size_t *> d_all_offsets("d_all_offsets",
                                             all_offsets.size());

        Kokkos::deep_copy(
            d_all_indices,
            UnmanagedSizeTHostView(all_indices.data(), all_indices.size()));
        Kokkos::deep_copy(
            d_all_offsets,
            UnmanagedSizeTHostView(all_offsets.data(), all_offsets.size()));
        Kokkos::deep_copy(d_sorted_ind_wires,
                          UnmanagedSizeTHostView(sorted_ind_wires.data(),
                                                 sorted_ind_wires.size()));

        const int num_all_indices =
            all_indices.size(); // int is required by Kokkos::MDRangePolicy
        const int num_all_offsets = all_offsets.size();

        MDPolicyType_2D mdpolicy_2d0({{0, 0}},
                                     {{num_all_indices, num_all_offsets}});

        Kokkos::parallel_for(
            "Set_Prob", mdpolicy_2d0,
            getSubProbFunctor<PrecisionT>(arr_data, d_probabilities,
                                          d_all_indices, d_all_offsets));

        std::vector<PrecisionT> probabilities(all_indices.size(), 0);

        if (is_sorted_wires) {
            Kokkos::deep_copy(UnmanagedPrecisionHostView(probabilities.data(),
                                                         probabilities.size()),
                              d_probabilities);
            return probabilities;
        } else {
            Kokkos::View<PrecisionT *> transposed_tensor("transposed_tensor",
                                                         all_indices.size());

            Kokkos::View<size_t *> d_trans_index("d_trans_index",
                                                 all_indices.size());

            const int num_trans_tensor = transposed_tensor.size();
            const int num_sorted_ind_wires = sorted_ind_wires.size();

            MDPolicyType_2D mdpolicy_2d1(
                {{0, 0}}, {{num_trans_tensor, num_sorted_ind_wires}});

            Kokkos::parallel_for(
                "TransIndex", mdpolicy_2d1,
                getTransposedIndexFunctor(d_sorted_ind_wires, d_trans_index,
                                          num_sorted_ind_wires));

            Kokkos::parallel_for(
                "Transpose",
                Kokkos::RangePolicy<KokkosExecSpace>(0, num_trans_tensor),
                getTransposedFunctor<PrecisionT>(
                    transposed_tensor, d_probabilities, d_trans_index));

            Kokkos::deep_copy(UnmanagedPrecisionHostView(probabilities.data(),
                                                         probabilities.size()),
                              transposed_tensor);

            return probabilities;
        }
    }

    /**
     * @brief  Inverse transform sampling method for samples.
     * Reference https://en.wikipedia.org/wiki/Inverse_transform_sampling
     *
     * @param num_samples Number of Samples
     *
     * @return std::vector<size_t> to the samples.
     * Each sample has a length equal to the number of qubits. Each sample can
     * be accessed using the stride sample_id*num_qubits, where sample_id is a
     * number between 0 and num_samples-1.
     */
    auto generate_samples(size_t num_samples) -> std::vector<size_t> {
        // printf("\nEnter generate_samples\n");
        const size_t num_qubits = this->_statevector.getNumQubits();
        const size_t N = this->_statevector.getLength();

        // printf("\nInit views\n");
        Kokkos::View<ComplexT *> arr_data = this->_statevector.getView();
        Kokkos::View<PrecisionT *> probability("probability", N);
        Kokkos::View<PrecisionT *> cumprob("cumprob", N);
        Kokkos::View<size_t *> samples("num_samples", num_samples * num_qubits);

        // printf("\ngetProb\n");
        Kokkos::fence();
        // Compute probability distribution from StateVector
        Kokkos::parallel_for(
            "getProb", Kokkos::RangePolicy<KokkosExecSpace>(0, N),
            //  getProbFunctor<PrecisionT>(arr_data, probability)
            KOKKOS_LAMBDA(const size_t k) {
                PrecisionT REAL = arr_data[k].real();
                PrecisionT IMAG = arr_data[k].imag();
                probability[k] = REAL * REAL + IMAG * IMAG;
            });
        Kokkos::fence();

        // printf("\ngetCDF\n");

        typename Kokkos::View<PrecisionT *>::HostMirror probability_h =
        Kokkos::create_mirror_view(probability);
        Kokkos::deep_copy(probability_h,probability);
        for (size_t k=1; k < N; k++){
            probability_h(k) += probability_h(k - 1);
        }
        Kokkos::deep_copy(probability,probability_h);

        // Convert probability distribution to cumulative distribution
        // Kokkos::parallel_scan(
        //     "getCDF", Kokkos::RangePolicy<KokkosExecSpace>(0, N),
        //     KOKKOS_LAMBDA(const size_t k, PrecisionT &update_value,
        //                   bool is_final) {
        //         update_value += probability(k);
        //         if (is_final)
        //             cumprob(k) = update_value;
        //     });

        // Sampling using Random_XorShift64_Pool
        // printf("\nInit rand_pool\n");
        Kokkos::Random_XorShift64_Pool<> rand_pool(5374857);

        // printf("\nSampler\n");
        Kokkos::parallel_for(
            Kokkos::RangePolicy<KokkosExecSpace>(0, num_samples),
            Sampler<PrecisionT, Kokkos::Random_XorShift64_Pool>(
                samples, probability, rand_pool, num_qubits, N));

        std::vector<size_t> samples_h(num_samples * num_qubits);

        using UnmanagedSize_tHostView =
            Kokkos::View<size_t *, Kokkos::HostSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

        // printf("\ndeep_copy\n");
        Kokkos::deep_copy(
            UnmanagedSize_tHostView(samples_h.data(), samples_h.size()),
            samples);

        return samples_h;
    }
};

} // namespace Pennylane::LightningKokkos::Measures

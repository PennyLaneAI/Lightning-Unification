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
 * @file
 * Minimal class for the Lightning qubit state vector interfacing with the
 * dynamic dispatcher and threading functionalities. This class is a bridge
 * between the base (agnostic) class and specializations for distinct data
 * storage types.
 */

#pragma once
#include <complex>
#include <unordered_map>

#include "CPUMemoryModel.hpp"
#include "GateOperation.hpp"
#include "KernelMap.hpp"
#include "KernelType.hpp"
#include "StateVectorBase.hpp"
#include "Threading.hpp"

/// @cond DEV
namespace {
using Pennylane::Util::CPUMemoryModel;
using Pennylane::Util::exp2;
using Pennylane::Util::Threading;

using namespace Pennylane::LightningQubit::Gates;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit {
/**
 * @brief Lightning qubit state vector class.
 *
 * Minimal class, without data storage, for the Lightning qubit state vector.
 * This class interfaces with the dynamic dispatcher and threading
 * functionalities and is a bridge between the base (agnostic) class and
 * specializations for distinct data storage types.
 *
 * @tparam PrecisionT Floating point precision of underlying state vector data.
 * @tparam Derived Derived class for CRTP.
 */
template <class PrecisionT, class Derived>
class StateVectorLQubit : public StateVectorBase<PrecisionT, Derived> {
  public:
    using ComplexPrecisionT = std::complex<PrecisionT>;

  protected:
    const Threading threading_;
    const CPUMemoryModel memory_model_;

  private:
    using BaseType = StateVectorBase<PrecisionT, Derived>;

    std::unordered_map<GateOperation, KernelType> kernel_for_gates_;
    std::unordered_map<GeneratorOperation, KernelType> kernel_for_generators_;
    std::unordered_map<MatrixOperation, KernelType> kernel_for_matrices_;

    /**
     * @brief Internal function to set kernels for all operations depending on
     * provided dispatch options.
     *
     * @param num_qubits Number of qubits of the statevector
     * @param threading Threading option
     * @param memory_model Memory model
     */
    void setKernels(size_t num_qubits, Threading threading,
                    CPUMemoryModel memory_model) {
        using KernelMap::OperationKernelMap;
        kernel_for_gates_ =
            OperationKernelMap<GateOperation>::getInstance().getKernelMap(
                num_qubits, threading, memory_model);
        kernel_for_generators_ =
            OperationKernelMap<GeneratorOperation>::getInstance().getKernelMap(
                num_qubits, threading, memory_model);
        kernel_for_matrices_ =
            OperationKernelMap<MatrixOperation>::getInstance().getKernelMap(
                num_qubits, threading, memory_model);
    }

  protected:
    explicit StateVectorLQubit(size_t num_qubits, Threading threading,
                               CPUMemoryModel memory_model)
        : BaseType(num_qubits), threading_{threading}, memory_model_{
                                                           memory_model} {
        setKernels(num_qubits, threading, memory_model);
    }

  public:
    /**
     * @brief Get the data pointer of the statevector
     *
     * @return The pointer to the data of statevector
     */
    [[nodiscard]] inline auto getData() -> decltype(auto) {
        return static_cast<Derived *>(this)->getData();
    }

    [[nodiscard]] inline auto getData() const -> decltype(auto) {
        return static_cast<const Derived *>(this)->getData();
    }

    /**
     * @brief Get a kernel for a gate operation.
     *
     * @param gate_op Gate operation
     * @return KernelType
     */
    [[nodiscard]] inline auto getKernelForGate(GateOperation gate_op) const
        -> KernelType {
        return kernel_for_gates_.at(gate_op);
    }

    /**
     * @brief Get a kernel for a generator operation.
     *
     * @param gen_op Generator operation
     * @return KernelType
     */
    [[nodiscard]] inline auto
    getKernelForGenerator(GeneratorOperation gen_op) const -> KernelType {
        return kernel_for_generators_.at(gen_op);
    }

    /**
     * @brief Get a kernel for a matrix operation.
     *
     * @param mat_op Matrix operation
     * @return KernelType
     */
    [[nodiscard]] inline auto getKernelForMatrix(MatrixOperation mat_op) const
        -> KernelType {
        return kernel_for_matrices_.at(mat_op);
    }

    /**
     * @brief Get memory model of the statevector.
     */
    [[nodiscard]] inline CPUMemoryModel memoryModel() const {
        return memory_model_;
    }

    /**
     * @brief Get threading mode of the statevector.
     */
    [[nodiscard]] inline Threading threading() const { return threading_; }

    /**
     * @brief Get kernels for all gate operations.
     */
    [[nodiscard]] inline auto getGateKernelMap()
        const & -> const std::unordered_map<GateOperation, KernelType> & {
        return kernel_for_gates_;
    }

    [[nodiscard]] inline auto
    getGateKernelMap() && -> std::unordered_map<GateOperation, KernelType> {
        return kernel_for_gates_;
    }

    /**
     * @brief Get kernels for all generator operations.
     */
    [[nodiscard]] inline auto getGeneratorKernelMap()
        const & -> const std::unordered_map<GeneratorOperation, KernelType> & {
        return kernel_for_generators_;
    }

    [[nodiscard]] inline auto getGeneratorKernelMap()
        && -> std::unordered_map<GeneratorOperation, KernelType> {
        return kernel_for_generators_;
    }

    /**
     * @brief Get kernels for all matrix operations.
     */
    [[nodiscard]] inline auto getMatrixKernelMap()
        const & -> const std::unordered_map<MatrixOperation, KernelType> & {
        return kernel_for_matrices_;
    }

    [[nodiscard]] inline auto
    getMatrixKernelMap() && -> std::unordered_map<MatrixOperation, KernelType> {
        return kernel_for_matrices_;
    }

    /**
     * @brief Apply a single gate to the state-vector using a given kernel.
     *
     * @param kernel Kernel to run the operation.
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Optional parameter list for parametric gates.
     */
    void applyOperation(Gates::KernelType kernel, const std::string &opName,
                        const std::vector<size_t> &wires, bool inverse = false,
                        const std::vector<PrecisionT> &params = {}) {
        auto *arr = getData();
        DynamicDispatcher<PrecisionT>::getInstance().applyOperation(
            kernel, arr, this->getNumQubits(), opName, wires, inverse, params);
    }

    /**
     * @brief Apply a single gate to the state-vector.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use inverse of gate.
     * @param params Optional parameter list for parametric gates.
     */
    void applyOperation(const std::string &opName,
                        const std::vector<size_t> &wires, bool inverse = false,
                        const std::vector<PrecisionT> &params = {}) {
        auto *arr = getData();
        auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
        const auto gate_op = dispatcher.strToGateOp(opName);
        dispatcher.applyOperation(getKernelForGate(gate_op), arr,
                                  this->getNumQubits(), gate_op, wires, inverse,
                                  params);
    }

    /**
     * @brief Apply multiple gates to the state-vector.
     *
     * @param ops Vector of gate names to be applied in order.
     * @param ops_wires Vector of wires on which to apply index-matched gate
     * name.
     * @param ops_inverse Indicates whether gate at matched index is to be
     * inverted.
     * @param ops_params Optional parameter data for index matched gates.
     */
    void
    applyOperations(const std::vector<std::string> &ops,
                    const std::vector<std::vector<size_t>> &ops_wires,
                    const std::vector<bool> &ops_inverse,
                    const std::vector<std::vector<PrecisionT>> &ops_params) {
        const size_t numOperations = ops.size();
        PL_ABORT_IF(
            numOperations != ops_wires.size(),
            "Invalid arguments: number of operations, wires, inverses, and "
            "parameters must all be equal");
        PL_ABORT_IF(
            numOperations != ops_inverse.size(),
            "Invalid arguments: number of operations, wires, inverses, and "
            "parameters must all be equal");
        PL_ABORT_IF(
            numOperations != ops_params.size(),
            "Invalid arguments: number of operations, wires, inverses, and "
            "parameters must all be equal");
        for (size_t i = 0; i < numOperations; i++) {
            applyOperation(ops[i], ops_wires[i], ops_inverse[i], ops_params[i]);
        }
    }

    /**
     * @brief Apply multiple gates to the state-vector.
     *
     * @param ops Vector of gate names to be applied in order.
     * @param ops_wires Vector of wires on which to apply index-matched gate
     * name.
     * @param ops_inverse Indicates whether gate at matched index is to be
     * inverted.
     */
    void applyOperations(const std::vector<std::string> &ops,
                         const std::vector<std::vector<size_t>> &ops_wires,
                         const std::vector<bool> &ops_inverse) {
        const size_t numOperations = ops.size();
        if (numOperations != ops_wires.size()) {
            PL_ABORT(
                "Invalid arguments: number of operations, wires, and inverses "
                "must all be equal");
        }
        if (numOperations != ops_inverse.size()) {
            PL_ABORT(
                "Invalid arguments: number of operations, wires and inverses"
                "must all be equal");
        }
        for (size_t i = 0; i < numOperations; i++) {
            applyOperation(ops[i], ops_wires[i], ops_inverse[i], {});
        }
    }

    /**
     * @brief Apply a single generator to the state-vector using a given kernel.
     *
     * @param kernel Kernel to run the operation.
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param adj Indicates whether to use adjoint of operator.
     */
    [[nodiscard]] inline auto applyGenerator(Gates::KernelType kernel,
                                             const std::string &opName,
                                             const std::vector<size_t> &wires,
                                             bool adj = false) -> PrecisionT {
        auto *arr = getData();
        return DynamicDispatcher<PrecisionT>::getInstance().applyGenerator(
            kernel, arr, this->getNumQubits(), opName, wires, adj);
    }

    /**
     * @brief Apply a single generator to the state-vector.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires the gate applies to.
     * @param adj Indicates whether to use adjoint of operator.
     */
    [[nodiscard]] auto applyGenerator(const std::string &opName,
                                      const std::vector<size_t> &wires,
                                      bool adj = false) -> PrecisionT {
        auto *arr = getData();
        const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
        const auto gen_op = dispatcher.strToGeneratorOp(opName);
        return dispatcher.applyGenerator(getKernelForGenerator(gen_op), arr,
                                         this->getNumQubits(), opName, wires,
                                         adj);
    }

    /**
     * @brief Apply a given matrix directly to the statevector using a given
     * kernel.
     *
     * @param kernel Kernel to run the operation
     * @param matrix Pointer to the array data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void applyMatrix(Gates::KernelType kernel,
                            const ComplexPrecisionT *matrix,
                            const std::vector<size_t> &wires,
                            bool inverse = false) {

        const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
        auto *arr = getData();

        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");

        dispatcher.applyMatrix(kernel, arr, this->getNumQubits(), matrix, wires,
                               inverse);
    }

    /**
     * @brief Apply a given matrix directly to the statevector using a given
     * kernel.
     *
     * @param kernel Kernel to run the operation
     * @param matrix Matrix data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void applyMatrix(Gates::KernelType kernel,
                            const std::vector<ComplexPrecisionT> &matrix,
                            const std::vector<size_t> &wires,
                            bool inverse = false) {

        PL_ABORT_IF(matrix.size() != exp2(2 * wires.size()),
                    "The size of matrix does not match with the given "
                    "number of wires");

        applyMatrix(kernel, matrix.data(), wires, inverse);
    }

    /**
     * @brief Apply a given matrix directly to the statevector using a
     * raw matrix pointer vector.
     *
     * @param matrix Pointer to the array data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void applyMatrix(const ComplexPrecisionT *matrix,
                            const std::vector<size_t> &wires,
                            bool inverse = false) {
        using Gates::MatrixOperation;

        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");

        const auto kernel = [n_wires = wires.size(), this]() {
            switch (n_wires) {
            case 1:
                return getKernelForMatrix(MatrixOperation::SingleQubitOp);
            case 2:
                return getKernelForMatrix(MatrixOperation::TwoQubitOp);
            default:
                return getKernelForMatrix(MatrixOperation::MultiQubitOp);
            }
        }();
        applyMatrix(kernel, matrix, wires, inverse);
    }

    /**
     * @brief Apply a given matrix directly to the statevector.
     *
     * @param matrix Matrix data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    template <typename Alloc>
    inline void applyMatrix(const std::vector<ComplexPrecisionT, Alloc> &matrix,
                            const std::vector<size_t> &wires,
                            bool inverse = false) {
        PL_ABORT_IF(matrix.size() != exp2(2 * wires.size()),
                    "The size of matrix does not match with the given "
                    "number of wires");

        applyMatrix(matrix.data(), wires, inverse);
    }
};
} // namespace Pennylane::LightningQubit
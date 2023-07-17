// Copyright 2022 Xanadu Quantum Technologies Inc.

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
 * @file StateVectorKokkos.hpp
 */

#pragma once
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "BitUtil.hpp" // isPerfectPowerOf2
#include "Error.hpp"
#include "GateFunctors.hpp"
#include "StateVectorBase.hpp"
#include "Util.hpp"

/// @cond DEV
namespace {
using Pennylane::Util::exp2;
using Pennylane::Util::isPerfectPowerOf2;
using Pennylane::Util::log2;
using namespace Pennylane::Lightning_Kokkos::Functors;
} // namespace
/// @endcond

namespace Pennylane::Lightning_Kokkos {

/**
 * @brief Kokkos functor for setting the state vector
 *
 * @tparam fp_t Floating point precision of underlying statevector data
 */
template <typename fp_t> struct setStateVectorFunctor {
    Kokkos::View<Kokkos::complex<fp_t> *> a;
    Kokkos::View<size_t *> indices;
    Kokkos::View<Kokkos::complex<fp_t> *> values;
    setStateVectorFunctor(Kokkos::View<Kokkos::complex<fp_t> *> a_,
                          const Kokkos::View<size_t *> indices_,
                          const Kokkos::View<Kokkos::complex<fp_t> *> values_)
        : a(a_), indices(indices_), values(values_) {}
    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t i) const { a(indices[i]) = values[i]; }
};

/**
 * @brief  Kokkos state vector class
 *
 * @tparam fp_t Floating-point precision type.
 */
template <class fp_t = double>
class StateVectorKokkos final
    : public StateVectorBase<fp_t, StateVectorKokkos<fp_t>> {

  private:
    using BaseType = StateVectorBase<fp_t, StateVectorKokkos<fp_t>>;

  public:
    using PrecisionT = fp_t;
    using ComplexT = Kokkos::complex<fp_t>;
    using KokkosExecSpace = Kokkos::DefaultExecutionSpace;
    using KokkosVector = Kokkos::View<ComplexT *>;
    using KokkosSizeTVector = Kokkos::View<size_t *>;
    using KokkosRangePolicy = Kokkos::RangePolicy<KokkosExecSpace>;
    using UnmanagedComplexHostView =
        Kokkos::View<ComplexT *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UnmanagedSizeTHostView =
        Kokkos::View<size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UnmanagedConstComplexHostView =
        Kokkos::View<const ComplexT *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UnmanagedConstSizeTHostView =
        Kokkos::View<const size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    StateVectorKokkos() = delete;
    StateVectorKokkos(size_t num_qubits,
                      const Kokkos::InitializationSettings &kokkos_args = {})
        : BaseType{num_qubits},
          gates_{// Identity
                 {"PauliX",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyPauliX(std::forward<decltype(wires)>(wires),
                                  std::forward<decltype(adjoint)>(adjoint),
                                  std::forward<decltype(params)>(params));
                  }},
                 {"PauliY",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyPauliY(std::forward<decltype(wires)>(wires),
                                  std::forward<decltype(adjoint)>(adjoint),
                                  std::forward<decltype(params)>(params));
                  }},
                 {"PauliZ",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyPauliZ(std::forward<decltype(wires)>(wires),
                                  std::forward<decltype(adjoint)>(adjoint),
                                  std::forward<decltype(params)>(params));
                  }},
                 {"Hadamard",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyHadamard(std::forward<decltype(wires)>(wires),
                                    std::forward<decltype(adjoint)>(adjoint),
                                    std::forward<decltype(params)>(params));
                  }},
                 {"S",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyS(std::forward<decltype(wires)>(wires),
                             std::forward<decltype(adjoint)>(adjoint),
                             std::forward<decltype(params)>(params));
                  }},
                 {"T",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyT(std::forward<decltype(wires)>(wires),
                             std::forward<decltype(adjoint)>(adjoint),
                             std::forward<decltype(params)>(params));
                  }},
                 {"RX",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyRX(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
                 {"RY",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyRY(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
                 {"RZ",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyRZ(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
                 {"PhaseShift",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyPhaseShift(std::forward<decltype(wires)>(wires),
                                      std::forward<decltype(adjoint)>(adjoint),
                                      std::forward<decltype(params)>(params));
                  }},
                 {"Rot",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyRot(std::forward<decltype(wires)>(wires),
                               std::forward<decltype(adjoint)>(adjoint),
                               std::forward<decltype(params)>(params));
                  }},
                 {"CY",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyCY(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
                 {"CZ",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyCZ(std::forward<decltype(wires)>(wires),
                              std::forward<decltype(adjoint)>(adjoint),
                              std::forward<decltype(params)>(params));
                  }},
                 {"CNOT",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyCNOT(std::forward<decltype(wires)>(wires),
                                std::forward<decltype(adjoint)>(adjoint),
                                std::forward<decltype(params)>(params));
                  }},
                 {"SWAP",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applySWAP(std::forward<decltype(wires)>(wires),
                                std::forward<decltype(adjoint)>(adjoint),
                                std::forward<decltype(params)>(params));
                  }},
                 {"ControlledPhaseShift",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyControlledPhaseShift(
                          std::forward<decltype(wires)>(wires),
                          std::forward<decltype(adjoint)>(adjoint),
                          std::forward<decltype(params)>(params));
                  }},
                 {"CRX",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyCRX(std::forward<decltype(wires)>(wires),
                               std::forward<decltype(adjoint)>(adjoint),
                               std::forward<decltype(params)>(params));
                  }},
                 {"CRY",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyCRY(std::forward<decltype(wires)>(wires),
                               std::forward<decltype(adjoint)>(adjoint),
                               std::forward<decltype(params)>(params));
                  }},
                 {"CRZ",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyCRZ(std::forward<decltype(wires)>(wires),
                               std::forward<decltype(adjoint)>(adjoint),
                               std::forward<decltype(params)>(params));
                  }},
                 {"CRot",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyCRot(std::forward<decltype(wires)>(wires),
                                std::forward<decltype(adjoint)>(adjoint),
                                std::forward<decltype(params)>(params));
                  }},
                 {"IsingXX",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyIsingXX(std::forward<decltype(wires)>(wires),
                                   std::forward<decltype(adjoint)>(adjoint),
                                   std::forward<decltype(params)>(params));
                  }},
                 {"IsingXY",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyIsingXY(std::forward<decltype(wires)>(wires),
                                   std::forward<decltype(adjoint)>(adjoint),
                                   std::forward<decltype(params)>(params));
                  }},

                 {"IsingYY",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyIsingYY(std::forward<decltype(wires)>(wires),
                                   std::forward<decltype(adjoint)>(adjoint),
                                   std::forward<decltype(params)>(params));
                  }},

                 {"IsingZZ",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyIsingZZ(std::forward<decltype(wires)>(wires),
                                   std::forward<decltype(adjoint)>(adjoint),
                                   std::forward<decltype(params)>(params));
                  }},
                 {"SingleExcitation",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applySingleExcitation(
                          std::forward<decltype(wires)>(wires),
                          std::forward<decltype(adjoint)>(adjoint),
                          std::forward<decltype(params)>(params));
                  }},
                 {"SingleExcitationMinus",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applySingleExcitationMinus(
                          std::forward<decltype(wires)>(wires),
                          std::forward<decltype(adjoint)>(adjoint),
                          std::forward<decltype(params)>(params));
                  }},
                 {"SingleExcitationPlus",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applySingleExcitationPlus(
                          std::forward<decltype(wires)>(wires),
                          std::forward<decltype(adjoint)>(adjoint),
                          std::forward<decltype(params)>(params));
                  }},
                 {"DoubleExcitation",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyDoubleExcitation(
                          std::forward<decltype(wires)>(wires),
                          std::forward<decltype(adjoint)>(adjoint),
                          std::forward<decltype(params)>(params));
                  }},
                 {"DoubleExcitationMinus",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyDoubleExcitationMinus(
                          std::forward<decltype(wires)>(wires),
                          std::forward<decltype(adjoint)>(adjoint),
                          std::forward<decltype(params)>(params));
                  }},
                 {"DoubleExcitationPlus",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyDoubleExcitationPlus(
                          std::forward<decltype(wires)>(wires),
                          std::forward<decltype(adjoint)>(adjoint),
                          std::forward<decltype(params)>(params));
                  }},
                 {"MultiRZ",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyMultiRZ(std::forward<decltype(wires)>(wires),
                                   std::forward<decltype(adjoint)>(adjoint),
                                   std::forward<decltype(params)>(params));
                  }},
                 {"CSWAP",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyCSWAP(std::forward<decltype(wires)>(wires),
                                 std::forward<decltype(adjoint)>(adjoint),
                                 std::forward<decltype(params)>(params));
                  }},
                 {"Toffoli",
                  [&](auto &&wires, auto &&adjoint, auto &&params) {
                      applyToffoli(std::forward<decltype(wires)>(wires),
                                   std::forward<decltype(adjoint)>(adjoint),
                                   std::forward<decltype(params)>(params));
                  }}},
          generator_{
              {"RX",
               [&](auto &&wires, auto &&adjoint, auto &&params) {
                   return applyGeneratorRX(
                       std::forward<decltype(wires)>(wires),
                       std::forward<decltype(adjoint)>(adjoint),
                       std::forward<decltype(params)>(params));
               }},
              {"RY",
               [&](auto &&wires, auto &&adjoint, auto &&params) {
                   return applyGeneratorRY(
                       std::forward<decltype(wires)>(wires),
                       std::forward<decltype(adjoint)>(adjoint),
                       std::forward<decltype(params)>(params));
               }},
              {"RZ",
               [&](auto &&wires, auto &&adjoint, auto &&params) {
                   return applyGeneratorRZ(
                       std::forward<decltype(wires)>(wires),
                       std::forward<decltype(adjoint)>(adjoint),
                       std::forward<decltype(params)>(params));
               }},
              {"ControlledPhaseShift",
               [&](auto &&wires, auto &&adjoint, auto &&params) {
                   return applyGeneratorControlledPhaseShift(
                       std::forward<decltype(wires)>(wires),
                       std::forward<decltype(adjoint)>(adjoint),
                       std::forward<decltype(params)>(params));
               }},
              {"CRX",
               [&](auto &&wires, auto &&adjoint, auto &&params) {
                   return applyGeneratorCRX(
                       std::forward<decltype(wires)>(wires),
                       std::forward<decltype(adjoint)>(adjoint),
                       std::forward<decltype(params)>(params));
               }},
              {"CRY",
               [&](auto &&wires, auto &&adjoint, auto &&params) {
                   return applyGeneratorCRY(
                       std::forward<decltype(wires)>(wires),
                       std::forward<decltype(adjoint)>(adjoint),
                       std::forward<decltype(params)>(params));
               }},
              {"CRZ",
               [&](auto &&wires, auto &&adjoint, auto &&params) {
                   return applyGeneratorCRZ(
                       std::forward<decltype(wires)>(wires),
                       std::forward<decltype(adjoint)>(adjoint),
                       std::forward<decltype(params)>(params));
               }},
              {"IsingXX",
               [&](auto &&wires, auto &&adjoint, auto &&params) {
                   return applyGeneratorIsingXX(
                       std::forward<decltype(wires)>(wires),
                       std::forward<decltype(adjoint)>(adjoint),
                       std::forward<decltype(params)>(params));
               }},
              {"IsingXY",
               [&](auto &&wires, auto &&adjoint, auto &&params) {
                   return applyGeneratorIsingXY(
                       std::forward<decltype(wires)>(wires),
                       std::forward<decltype(adjoint)>(adjoint),
                       std::forward<decltype(params)>(params));
               }},
              {"IsingYY",
               [&](auto &&wires, auto &&adjoint, auto &&params) {
                   return applyGeneratorIsingYY(
                       std::forward<decltype(wires)>(wires),
                       std::forward<decltype(adjoint)>(adjoint),
                       std::forward<decltype(params)>(params));
               }},
              {"IsingZZ",
               [&](auto &&wires, auto &&adjoint, auto &&params) {
                   return applyGeneratorIsingZZ(
                       std::forward<decltype(wires)>(wires),
                       std::forward<decltype(adjoint)>(adjoint),
                       std::forward<decltype(params)>(params));
               }},
              {"SingleExcitation",
               [&](auto &&wires, auto &&adjoint, auto &&params) {
                   return applyGeneratorSingleExcitation(
                       std::forward<decltype(wires)>(wires),
                       std::forward<decltype(adjoint)>(adjoint),
                       std::forward<decltype(params)>(params));
               }},
              {"SingleExcitationMinus",
               [&](auto &&wires, auto &&adjoint, auto &&params) {
                   return applyGeneratorSingleExcitationMinus(
                       std::forward<decltype(wires)>(wires),
                       std::forward<decltype(adjoint)>(adjoint),
                       std::forward<decltype(params)>(params));
               }},
              {"SingleExcitationPlus",
               [&](auto &&wires, auto &&adjoint, auto &&params) {
                   return applyGeneratorSingleExcitationPlus(
                       std::forward<decltype(wires)>(wires),
                       std::forward<decltype(adjoint)>(adjoint),
                       std::forward<decltype(params)>(params));
               }},
              {"DoubleExcitation",
               [&](auto &&wires, auto &&adjoint, auto &&params) {
                   return applyGeneratorDoubleExcitation(
                       std::forward<decltype(wires)>(wires),
                       std::forward<decltype(adjoint)>(adjoint),
                       std::forward<decltype(params)>(params));
               }},
              {"DoubleExcitationMinus",
               [&](auto &&wires, auto &&adjoint, auto &&params) {
                   return applyGeneratorDoubleExcitationMinus(
                       std::forward<decltype(wires)>(wires),
                       std::forward<decltype(adjoint)>(adjoint),
                       std::forward<decltype(params)>(params));
               }},
              {"DoubleExcitationPlus",
               [&](auto &&wires, auto &&adjoint, auto &&params) {
                   return applyGeneratorDoubleExcitationPlus(
                       std::forward<decltype(wires)>(wires),
                       std::forward<decltype(adjoint)>(adjoint),
                       std::forward<decltype(params)>(params));
               }},
              {"PhaseShift",
               [&](auto &&wires, auto &&adjoint, auto &&params) {
                   return applyGeneratorPhaseShift(
                       std::forward<decltype(wires)>(wires),
                       std::forward<decltype(adjoint)>(adjoint),
                       std::forward<decltype(params)>(params));
               }},
              {"MultiRZ",
               [&](auto &&wires, auto &&adjoint, auto &&params) {
                   return applyGeneratorMultiRZ(
                       std::forward<decltype(wires)>(wires),
                       std::forward<decltype(adjoint)>(adjoint),
                       std::forward<decltype(params)>(params));
               }},
          } {
        num_qubits_ = num_qubits;
        length_ = exp2(num_qubits);

        {
            const std::lock_guard<std::mutex> lock(init_mutex_);
            if (!Kokkos::is_initialized()) {
                Kokkos::initialize(kokkos_args);
            }
        }

        if (num_qubits > 0) {
            data_ = std::make_unique<KokkosVector>("data_", exp2(num_qubits));
            setBasisState(0U);
        }
    };

    /**
     * @brief Init zeros for the state-vector on device.
     */
    void initZeros() { Kokkos::deep_copy(getData(), ComplexT(0.0, 0.0)); }

    /**
     * @brief Set value for a single element of the state-vector on device.
     *
     * @param index Index of the target element.
     */
    void setBasisState(const size_t index) {
        initZeros();
        getData()(index) = ComplexT(1.0, 0.0);
    }

    /**
     * @brief Set values for a batch of elements of the state-vector.
     *
     * @param values Values to be set for the target elements.
     * @param indices Indices of the target elements.
     */
    void setStateVector(const std::vector<std::size_t> &indices,
                        const std::vector<ComplexT> &values) {

        initZeros();

        KokkosSizeTVector d_indices("d_indices", indices.size());

        KokkosVector d_values("d_values", values.size());

        Kokkos::deep_copy(d_indices, UnmanagedConstSizeTHostView(
                                         indices.data(), indices.size()));

        Kokkos::deep_copy(d_values, UnmanagedConstComplexHostView(
                                        values.data(), values.size()));

        Kokkos::parallel_for(
            indices.size(),
            setStateVectorFunctor(getData(), d_indices, d_values));
    }

    /**
     * @brief Reset the data back to the \f$\ket{0}\f$ state.
     *
     * @param num_qubits Number of qubits
     */
    void resetStateVector() {
        if (length_ > 0) {
            setBasisState(0U);
        }
    }

    /**
     * @brief Create a new state vector from data on the host.
     *
     * @param num_qubits Number of qubits
     */
    StateVectorKokkos(ComplexT *hostdata_, size_t length,
                      const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkos(log2(length), kokkos_args) {
        PL_ABORT_IF_NOT(isPerfectPowerOf2(length),
                        "The size of provided data must be a power of 2.");
        HostToDevice(hostdata_, length);
    }

    /**
     * @brief Create a new state vector from data on the host.
     *
     * @param num_qubits Number of qubits
     */
    StateVectorKokkos(std::vector<ComplexT> hostdata_,
                      const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkos(hostdata_.data(), hostdata_.size(), kokkos_args) {}

    /**
     * @brief Copy constructor
     *
     * @param other Another state vector
     */
    StateVectorKokkos(const StateVectorKokkos &other,
                      const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorKokkos(other.getNumQubits(), kokkos_args) {
        this->DeviceToDevice(other.getData());
    }

    /**
     * @brief Destructor for StateVectorKokkos class
     *
     * @param other Another state vector
     */
    ~StateVectorKokkos() {
        data_.reset();
        {
            const std::lock_guard<std::mutex> lock(init_mutex_);
            if (!is_exit_reg_) {
                is_exit_reg_ = true;
                std::atexit([]() {
                    if (!Kokkos::is_finalized()) {
                        Kokkos::finalize();
                    }
                });
            }
        }
    }

    /**
     * @brief Apply a single gate to the state vector.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param adjoint Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     * @param gate_matrix Optional gate matrix if opName doesn't exist
     */
    void applyOperation(const std::string &opName,
                        const std::vector<size_t> &wires, bool adjoint = false,
                        const std::vector<fp_t> &params = {0.0},
                        [[maybe_unused]] const KokkosVector &gate_matrix = {}) {
        if (opName == "Identity") {
            // No op
        } else if (gates_.find(opName) != gates_.end()) {
            gates_.at(opName)(wires, adjoint, params);
        } else {
            KokkosVector matrix("gate_matrix", gate_matrix.size());
            Kokkos::deep_copy(matrix,
                              UnmanagedComplexHostView(gate_matrix.data(),
                                                       gate_matrix.size()));
            return applyMultiQubitOp(matrix, wires, adjoint);
        }
    }

    /**
     * @brief Apply a single gate to the state vector.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param adjoint Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     * @param params Optional std gate matrix if opName doesn't exist.
     */
    void applyOperation_std(
        const std::string &opName, const std::vector<size_t> &wires,
        bool adjoint = false, const std::vector<fp_t> &params = {0.0},
        [[maybe_unused]] const std::vector<ComplexT> &gate_matrix = {}) {

        if (opName == "Identity") {
            // No op
        } else if (gates_.find(opName) != gates_.end()) {
            gates_.at(opName)(wires, adjoint, params);
        } else {
            KokkosVector matrix("gate_matrix", gate_matrix.size());
            Kokkos::deep_copy(
                matrix, UnmanagedConstComplexHostView(gate_matrix.data(),
                                                      gate_matrix.size()));
            return applyMultiQubitOp(matrix, wires, adjoint);
        }
    }

    /**
     * @brief Apply a single generator to the state vector using the given
     * kernel.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param adjoint Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     */
    auto applyGenerator(const std::string &opName,
                        const std::vector<size_t> &wires, bool adjoint = false,
                        const std::vector<fp_t> &params = {0.0}) -> fp_t {
        const auto it = generator_.find(opName);
        PL_ABORT_IF(it == generator_.end(),
                    std::string("Generator does not exist for ") + opName);
        return (it->second)(wires, adjoint, params);
    }

    /**
     * @brief Apply a single qubit operator to the state vector using a matrix
     *
     * @param matrix Kokkos gate matrix in the device space
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     */
    void applySingleQubitOp(const KokkosVector &matrix,
                            const std::vector<size_t> &wires,
                            bool inverse = false) {
        auto &&num_qubits = this->getNumQubits();
        if (!inverse) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(0, exp2(num_qubits - 1)),
                singleQubitOpFunctor<fp_t, false>(*data_, num_qubits, matrix,
                                                  wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(0, exp2(num_qubits - 1)),
                singleQubitOpFunctor<fp_t, true>(*data_, num_qubits, matrix,
                                                 wires));
        }
    }

    /**
     * @brief Apply a two qubit operator to the state vector using a matrix
     *
     * @param matrix Kokkos gate matrix in the device space
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     */
    void applyTwoQubitOp(const KokkosVector &matrix,
                         const std::vector<size_t> &wires,
                         bool inverse = false) {

        auto &&num_qubits = this->getNumQubits();
        if (!inverse) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(0, exp2(num_qubits - 2)),
                twoQubitOpFunctor<fp_t, false>(*data_, num_qubits, matrix,
                                               wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(0, exp2(num_qubits - 2)),
                twoQubitOpFunctor<fp_t, true>(*data_, num_qubits, matrix,
                                              wires));
        }
    }

    /**
     * @brief Apply a multi qubit operator to the state vector using a matrix
     *
     * @param matrix Kokkos gate matrix in the device space
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     */
    void applyMultiQubitOp(const KokkosVector &matrix,
                           const std::vector<size_t> &wires,
                           bool inverse = false) {
        auto &&num_qubits = this->getNumQubits();
        if (wires.size() == 1) {
            applySingleQubitOp(matrix, wires, inverse);
        } else if (wires.size() == 2) {
            applyTwoQubitOp(matrix, wires, inverse);
        } else {

            Kokkos::View<const std::size_t *, Kokkos::HostSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                wires_host(wires.data(), wires.size());

            Kokkos::View<std::size_t *> wires_view("wires_view", wires.size());
            Kokkos::deep_copy(wires_view, wires_host);

            if (!inverse) {
                Kokkos::parallel_for(
                    Kokkos::RangePolicy<KokkosExecSpace>(
                        0, exp2(num_qubits_ - wires.size())),
                    multiQubitOpFunctor<fp_t, false>(*data_, num_qubits, matrix,
                                                     wires_view));
            } else {
                Kokkos::parallel_for(
                    Kokkos::RangePolicy<KokkosExecSpace>(
                        0, exp2(num_qubits_ - wires.size())),
                    multiQubitOpFunctor<fp_t, true>(*data_, num_qubits, matrix,
                                                    wires_view));
            }
        }
    }

    /**
     * @brief Apply a multi qubit operator to the state vector using a matrix
     *
     * @param matrix Kokkos gate matrix in the device space
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     */
    inline void applyMatrix(const KokkosVector &matrix,
                            const std::vector<size_t> &wires,
                            bool inverse = false) {
        applyMultiQubitOp(matrix, wires, inverse);
    }

    /**
     * @brief Apply a given matrix directly to the statevector using a
     * raw matrix pointer vector.
     *
     * @param matrix Pointer to the array data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void applyMatrix(ComplexT *matrix, const std::vector<size_t> &wires,
                            bool inverse = false) {
        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");
        size_t n = 1U << wires.size();
        KokkosVector matrix_(matrix, n * n);
        applyMultiQubitOp(matrix_, wires, inverse);
    }

    /**
     * @brief Apply a given matrix directly to the statevector.
     *
     * @param matrix Matrix data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void applyMatrix(std::vector<ComplexT> &matrix,
                            const std::vector<size_t> &wires,
                            bool inverse = false) {
        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");
        PL_ABORT_IF(matrix.size() != exp2(2 * wires.size()),
                    "The size of matrix does not match with the given "
                    "number of wires");
        applyMatrix(matrix.data(), wires, inverse);
    }

    /**
     * @brief Apply a given matrix directly to the statevector using a
     * raw matrix pointer vector.
     *
     * @param matrix Pointer to the array data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void applyMatrix(const ComplexT *matrix,
                            const std::vector<size_t> &wires,
                            bool inverse = false) {
        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");
        size_t n = 1U << wires.size();
        KokkosVector matrix_("matrix_", n * n);
        for (size_t i = 0; i < n * n; i++) {
            matrix_(i) = matrix[i];
        }
        applyMultiQubitOp(matrix_, wires, inverse);
    }

    /**
     * @brief Apply a given matrix directly to the statevector.
     *
     * @param matrix Matrix data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void applyMatrix(const std::vector<ComplexT> &matrix,
                            const std::vector<size_t> &wires,
                            bool inverse = false) {
        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");
        PL_ABORT_IF(matrix.size() != exp2(2 * wires.size()),
                    "The size of matrix does not match with the given "
                    "number of wires");
        applyMatrix(matrix.data(), wires, inverse);
    }

    /**
     * @brief Templated method that applies special n-qubit gates.
     *
     * @tparam functor_t Gate functor class for Kokkos dispatcher.
     * @tparam nqubits Number of qubits.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    template <template <class, bool> class functor_t, int nqubits>
    void
    applyGateFunctor(const std::vector<size_t> &wires, bool inverse = false,
                     [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        auto &&num_qubits = this->getNumQubits();
        PL_ASSERT(wires.size() == nqubits);
        if (!inverse) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, exp2(num_qubits - nqubits)),
                functor_t<fp_t, false>(*data_, num_qubits, wires, params));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, exp2(num_qubits - nqubits)),
                functor_t<fp_t, true>(*data_, num_qubits, wires, params));
        }
    }

    /**
     * @brief Apply a PauliX operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params Parameters for this gate
     */
    void applyPauliX(const std::vector<size_t> &wires, bool inverse = false,
                     [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<pauliXFunctor, 1>(wires, inverse, params);
    }

    /**
     * @brief Apply a PauliY operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params Parameters for this gate
     */
    void applyPauliY(const std::vector<size_t> &wires, bool inverse = false,
                     [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<pauliYFunctor, 1>(wires, inverse, params);
    }

    /**
     * @brief Apply a PauliZ operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params Parameters for this gate
     */

    void applyPauliZ(const std::vector<size_t> &wires, bool inverse = false,
                     [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<pauliZFunctor, 1>(wires, inverse, params);
    }

    /**
     * @brief Apply a Hadamard operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyHadamard(const std::vector<size_t> &wires, bool inverse = false,
                       [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<hadamardFunctor, 1>(wires, inverse, params);
    }

    /**
     * @brief Apply a S operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyS(const std::vector<size_t> &wires, bool inverse = false,
                [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<sFunctor, 1>(wires, inverse, params);
    }

    /**
     * @brief Apply a T operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyT(const std::vector<size_t> &wires, bool inverse = false,
                [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<tFunctor, 1>(wires, inverse, params);
    }

    /**
     * @brief Apply a RX operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyRX(const std::vector<size_t> &wires, bool inverse,
                 [[maybe_unused]] const std::vector<fp_t> &params) {
        applyGateFunctor<rxFunctor, 1>(wires, inverse, params);
    }

    /**
     * @brief Apply a RY operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyRY(const std::vector<size_t> &wires, bool inverse,
                 [[maybe_unused]] const std::vector<fp_t> &params) {
        applyGateFunctor<ryFunctor, 1>(wires, inverse, params);
    }

    /**
     * @brief Apply a RZ operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyRZ(const std::vector<size_t> &wires, bool inverse = false,
                 [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<rzFunctor, 1>(wires, inverse, params);
    }

    /**
     * @brief Apply a PhaseShift operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void
    applyPhaseShift(const std::vector<size_t> &wires, bool inverse = false,
                    [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<phaseShiftFunctor, 1>(wires, inverse, params);
    }

    /**
     * @brief Apply a Rot operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyRot(const std::vector<size_t> &wires, bool inverse,
                  const std::vector<fp_t> &params) {
        applyGateFunctor<rotFunctor, 1>(wires, inverse, params);
    }

    /**
     * @brief Apply a CY operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyCY(const std::vector<size_t> &wires, bool inverse = false,
                 [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<cyFunctor, 2>(wires, inverse, params);
    }

    /**
     * @brief Apply a CZ operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyCZ(const std::vector<size_t> &wires, bool inverse = false,
                 [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<czFunctor, 2>(wires, inverse, params);
    }

    /**
     * @brief Apply a CNOT operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyCNOT(const std::vector<size_t> &wires, bool inverse = false,
                   [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<cnotFunctor, 2>(wires, inverse, params);
    }

    /**
     * @brief Apply a SWAP operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applySWAP(const std::vector<size_t> &wires, bool inverse = false,
                   [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<swapFunctor, 2>(wires, inverse, params);
    }

    /**
     * @brief Apply a ControlledPhaseShift operator to the state vector using a
     * matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyControlledPhaseShift(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<controlledPhaseShiftFunctor, 2>(wires, inverse,
                                                         params);
    }

    /**
     * @brief Apply a CRX operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyCRX(const std::vector<size_t> &wires, bool inverse = false,
                  [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<crxFunctor, 2>(wires, inverse, params);
    }

    /**
     * @brief Apply a CRY operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyCRY(const std::vector<size_t> &wires, bool inverse = false,
                  [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<cryFunctor, 2>(wires, inverse, params);
    }

    /**
     * @brief Apply a CRZ operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyCRZ(const std::vector<size_t> &wires, bool inverse = false,
                  [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<crzFunctor, 2>(wires, inverse, params);
    }

    /**
     * @brief Apply a CRot operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyCRot(const std::vector<size_t> &wires, bool inverse,
                   [[maybe_unused]] const std::vector<fp_t> &params) {
        applyGateFunctor<cRotFunctor, 2>(wires, inverse, params);
    }

    /**
     * @brief Apply a IsingXX operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyIsingXX(const std::vector<size_t> &wires, bool inverse = false,
                      [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<isingXXFunctor, 2>(wires, inverse, params);
    }

    /**
     * @brief Apply a IsingXY operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyIsingXY(const std::vector<size_t> &wires, bool inverse = false,
                      [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<isingXYFunctor, 2>(wires, inverse, params);
    }

    /**
     * @brief Apply a IsingYY operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyIsingYY(const std::vector<size_t> &wires, bool inverse = false,
                      [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<isingYYFunctor, 2>(wires, inverse, params);
    }

    /**
     * @brief Apply a IsingZZ operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyIsingZZ(const std::vector<size_t> &wires, bool inverse = false,
                      [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<isingZZFunctor, 2>(wires, inverse, params);
    }

    /**
     * @brief Apply a SingleExcitation operator to the state vector using a
     * matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applySingleExcitation(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<singleExcitationFunctor, 2>(wires, inverse, params);
    }

    /**
     * @brief Apply a SingleExcitationMinus operator to the state vector using a
     * matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applySingleExcitationMinus(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<singleExcitationMinusFunctor, 2>(wires, inverse,
                                                          params);
    }

    /**
     * @brief Apply a SingleExcitationPlus operator to the state vector using a
     * matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applySingleExcitationPlus(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<singleExcitationPlusFunctor, 2>(wires, inverse,
                                                         params);
    }

    /**
     * @brief Apply a DoubleExcitation operator to the state vector using a
     * matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyDoubleExcitation(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<doubleExcitationFunctor, 4>(wires, inverse, params);
    }

    /**
     * @brief Apply a DoubleExcitationMinus operator to the state vector using a
     * matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyDoubleExcitationMinus(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<doubleExcitationMinusFunctor, 4>(wires, inverse,
                                                          params);
    }

    /**
     * @brief Apply a DoubleExcitationPlus operator to the state vector using a
     * matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyDoubleExcitationPlus(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<doubleExcitationPlusFunctor, 4>(wires, inverse,
                                                         params);
    }

    /**
     * @brief Apply a MultiRZ operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyMultiRZ(const std::vector<size_t> &wires, bool inverse = false,
                      [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        auto &&num_qubits = this->getNumQubits();

        if (!inverse) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(0, exp2(num_qubits)),
                multiRZFunctor<fp_t, false>(*data_, num_qubits, wires, params));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(0, exp2(num_qubits)),
                multiRZFunctor<fp_t, true>(*data_, num_qubits, wires, params));
        }
    }

    /**
     * @brief Apply a CSWAP operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyCSWAP(const std::vector<size_t> &wires, bool inverse = false,
                    [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<cSWAPFunctor, 3>(wires, inverse, params);
    }

    /**
     * @brief Apply a Toffoli operator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    void applyToffoli(const std::vector<size_t> &wires, bool inverse = false,
                      [[maybe_unused]] const std::vector<fp_t> &params = {}) {
        applyGateFunctor<toffoliFunctor, 3>(wires, inverse, params);
    }

    /**
     * @brief Apply a PhaseShift generator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorPhaseShift(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<fp_t> &params = {}) -> fp_t {
        applyGateFunctor<generatorPhaseShiftFunctor, 1>(wires, inverse, params);
        return static_cast<fp_t>(1.0);
    }

    /**
     * @brief Apply a IsingXX generator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorIsingXX(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<fp_t> &params = {}) -> fp_t {
        applyGateFunctor<generatorIsingXXFunctor, 2>(wires, inverse, params);
        return -static_cast<fp_t>(0.5);
    }

    /**
     * @brief Apply a IsingXY generator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorIsingXY(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<fp_t> &params = {}) -> fp_t {
        applyGateFunctor<generatorIsingXYFunctor, 2>(wires, inverse, params);
        return static_cast<fp_t>(0.5);
    }

    /**
     * @brief Apply a IsingYY generator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorIsingYY(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<fp_t> &params = {}) -> fp_t {
        applyGateFunctor<generatorIsingYYFunctor, 2>(wires, inverse, params);
        return -static_cast<fp_t>(0.5);
    }

    /**
     * @brief Apply a IsingZZ generator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorIsingZZ(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<fp_t> &params = {}) -> fp_t {
        applyGateFunctor<generatorIsingZZFunctor, 2>(wires, inverse, params);
        return -static_cast<fp_t>(0.5);
    }

    /**
     * @brief Apply a SingleExcitation generator to the state vector using a
     * matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorSingleExcitation(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<fp_t> &params = {}) -> fp_t {
        applyGateFunctor<generatorSingleExcitationFunctor, 2>(wires, inverse,
                                                              params);
        return -static_cast<fp_t>(0.5);
    }

    /**
     * @brief Apply a SingleExcitationMinus generator to the state vector using
     * a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorSingleExcitationMinus(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<fp_t> &params = {}) -> fp_t {
        applyGateFunctor<generatorSingleExcitationMinusFunctor, 2>(
            wires, inverse, params);
        return -static_cast<fp_t>(0.5);
    }

    /**
     * @brief Apply a SingleExcitationPlus generator to the state vector using a
     * matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorSingleExcitationPlus(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<fp_t> &params = {}) -> fp_t {
        applyGateFunctor<generatorSingleExcitationPlusFunctor, 2>(
            wires, inverse, params);
        return -static_cast<fp_t>(0.5);
    }

    /**
     * @brief Apply a DoubleExcitation generator to the state vector using a
     * matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorDoubleExcitation(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<fp_t> &params = {}) -> fp_t {
        applyGateFunctor<generatorDoubleExcitationFunctor, 4>(wires, inverse,
                                                              params);
        return -static_cast<fp_t>(0.5);
    }

    /**
     * @brief Apply a DoubleExcitationMinus generator to the state vector using
     * a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorDoubleExcitationMinus(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<fp_t> &params = {}) -> fp_t {
        applyGateFunctor<generatorDoubleExcitationMinusFunctor, 4>(
            wires, inverse, params);
        return -static_cast<fp_t>(0.5);
    }

    /**
     * @brief Apply a DoubleExcitationPlus generator to the state vector using a
     * matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorDoubleExcitationPlus(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<fp_t> &params = {}) -> fp_t {
        applyGateFunctor<generatorDoubleExcitationPlusFunctor, 4>(
            wires, inverse, params);
        return static_cast<fp_t>(0.5);
    }

    /**
     * @brief Apply a RX generator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorRX(const std::vector<size_t> &wires,
                          bool inverse = false,
                          [[maybe_unused]] const std::vector<fp_t> &params = {})
        -> fp_t {
        applyPauliX(wires, inverse, params);
        return -static_cast<fp_t>(0.5);
    }

    /**
     * @brief Apply a RY generator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorRY(const std::vector<size_t> &wires,
                          bool inverse = false,
                          [[maybe_unused]] const std::vector<fp_t> &params = {})
        -> fp_t {
        applyPauliY(wires, inverse, params);
        return -static_cast<fp_t>(0.5);
    }

    /**
     * @brief Apply a RZ generator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorRZ(const std::vector<size_t> &wires,
                          bool inverse = false,
                          [[maybe_unused]] const std::vector<fp_t> &params = {})
        -> fp_t {
        applyPauliZ(wires, inverse, params);
        return -static_cast<fp_t>(0.5);
    }

    /**
     * @brief Apply a ControlledPhaseShift generator to the state vector using a
     * matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorControlledPhaseShift(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<fp_t> &params = {}) -> fp_t {
        applyGateFunctor<generatorControlledPhaseShiftFunctor, 2>(
            wires, inverse, params);
        return static_cast<fp_t>(1);
    }

    /**
     * @brief Apply a CRX generator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto
    applyGeneratorCRX(const std::vector<size_t> &wires, bool inverse = false,
                      [[maybe_unused]] const std::vector<fp_t> &params = {})

        -> fp_t {
        applyGateFunctor<generatorCRXFunctor, 2>(wires, inverse, params);
        return -static_cast<fp_t>(0.5);
    }

    /**
     * @brief Apply a CRY generator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto
    applyGeneratorCRY(const std::vector<size_t> &wires, bool inverse = false,
                      [[maybe_unused]] const std::vector<fp_t> &params = {})
        -> fp_t {
        applyGateFunctor<generatorCRYFunctor, 2>(wires, inverse, params);
        return -static_cast<fp_t>(0.5);
    }

    /**
     * @brief Apply a CRZ generator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto
    applyGeneratorCRZ(const std::vector<size_t> &wires, bool inverse = false,
                      [[maybe_unused]] const std::vector<fp_t> &params = {})
        -> fp_t {
        applyGateFunctor<generatorCRZFunctor, 2>(wires, inverse, params);
        return -static_cast<fp_t>(0.5);
    }

    /**
     * @brief Apply a MultiRZ generator to the state vector using a matrix
     *
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params parameters for this gate
     */
    auto applyGeneratorMultiRZ(
        const std::vector<size_t> &wires, bool inverse = false,
        [[maybe_unused]] const std::vector<fp_t> &params = {}) -> fp_t {
        auto &&num_qubits = this->getNumQubits();

        if (inverse == false) {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(0, exp2(num_qubits)),
                generatorMultiRZFunctor<fp_t, false>(*data_, num_qubits,
                                                     wires));
        } else {
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(0, exp2(num_qubits)),
                generatorMultiRZFunctor<fp_t, true>(*data_, num_qubits, wires));
        }
        return -static_cast<fp_t>(0.5);
    }

    /**
     * @brief Get the number of qubits of the state vector.
     *
     * @return The number of qubits of the state vector
     */
    size_t getNumQubits() const { return num_qubits_; }

    /**
     * @brief Get the size of the state vector
     *
     * @return The size of the state vector
     */
    size_t getLength() const { return length_; }

    /**
     * @brief Update data of the class
     *
     * @param other Kokkos View
     */
    void updateData(const KokkosVector &other) {
        Kokkos::deep_copy(*data_, other);
    }

    /**
     * @brief Update data of the class
     *
     * @param other State vector
     */
    void updateData(const StateVectorKokkos<fp_t> &other) {
        updateData(other.getData());
    }

    /**
     * @brief Update data of the class
     *
     * @param new_data data pointer to new data.
     * @param new_size size of underlying data storage.
     */
    void updateData(ComplexT *new_data, size_t new_size) {
        updateData(KokkosVector(new_data, new_size));
    }

    /**
     * @brief Update data of the class
     *
     * @param other STL vector of type ComplexT
     */
    void updateData(std::vector<ComplexT> &other) {
        updateData(other.data(), other.size());
    }

    /**
     * @brief Get the Kokkos data of the state vector.
     *
     * @return The pointer to the data of state vector
     */
    [[nodiscard]] auto getData() const -> KokkosVector & { return *data_; }

    /**
     * @brief Get the Kokkos data of the state vector
     *
     * @return The pointer to the data of state vector
     */
    [[nodiscard]] auto getData() -> KokkosVector & { return *data_; }

    /**
     * @brief Get underlying data vector
     */
    [[nodiscard]] auto getDataVector() -> std::vector<ComplexT> {
        std::vector<ComplexT> data_(getData().data(),
                                    getData().data() + getData().size());
        return data_;
    }

    [[nodiscard]] auto getDataVector() const -> const std::vector<ComplexT> {
        const std::vector<ComplexT> data_(getData().data(),
                                          getData().data() + getData().size());
        return data_;
    }

    /**
     * @brief Copy data from the host space to the device space.
     *
     */
    inline void HostToDevice(ComplexT *sv, size_t length) {
        Kokkos::deep_copy(*data_, UnmanagedComplexHostView(sv, length));
    }

    /**
     * @brief Copy data from the device space to the host space.
     *
     */
    inline void DeviceToHost(ComplexT *sv, size_t length) {
        Kokkos::deep_copy(UnmanagedComplexHostView(sv, length), *data_);
    }

    /**
     * @brief Copy data from the device space to the device space.
     *
     */
    inline void DeviceToDevice(KokkosVector vector_to_copy) {
        Kokkos::deep_copy(*data_, vector_to_copy);
    }

  private:
    using GateFunc = std::function<void(const std::vector<size_t> &, bool,
                                        const std::vector<fp_t> &)>;
    using GateMap = std::unordered_map<std::string, GateFunc>;
    const GateMap gates_;

    using GeneratorFunc = std::function<fp_t(const std::vector<size_t> &, bool,
                                             const std::vector<fp_t> &)>;
    using GeneratorMap = std::unordered_map<std::string, GeneratorFunc>;
    const GeneratorMap generator_;

    size_t num_qubits_;
    size_t length_;
    std::mutex init_mutex_;
    std::unique_ptr<KokkosVector> data_;
    inline static bool is_exit_reg_ = false;
};

}; // namespace Pennylane::Lightning_Kokkos

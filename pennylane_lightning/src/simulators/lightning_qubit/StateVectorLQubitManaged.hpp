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
 * Statevector simulator where data management resides inside the class.
 */

#pragma once

#include <complex>
#include <vector>

#include "BitUtil.hpp"        // log2PerfectPower, isPerfectPowerOf2
#include "CPUMemoryModel.hpp" // bestCPUMemoryModel
#include "Gates.hpp"
#include "KernelType.hpp"
#include "Memory.hpp"
#include "StateVectorLQubit.hpp"
#include "Threading.hpp"
#include "Util.hpp" // exp2

/// @cond DEV
namespace {
using Pennylane::Util::AlignedAllocator;
using Pennylane::Util::bestCPUMemoryModel;
using Pennylane::Util::exp2;
using Pennylane::Util::isPerfectPowerOf2;
using Pennylane::Util::log2PerfectPower;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit {

/**
 * @brief StateVector class where data resides in CPU memory. Memory ownership
 * resides within class.
 *
 * @tparam PrecisionT Precision data type
 */
template <class PrecisionT = double>
class StateVectorLQubitManaged
    : public StateVectorLQubit<PrecisionT,
                               StateVectorLQubitManaged<PrecisionT>> {
  public:
    using BaseType = StateVectorLQubit<PrecisionT, StateVectorLQubitManaged>;
    using ComplexPrecisionT = std::complex<PrecisionT>;

  private:
    std::vector<ComplexPrecisionT, AlignedAllocator<ComplexPrecisionT>> data_;

  public:
    /**
     * @brief Create a new statevector in the computational basis state |0...0>
     *
     * @param num_qubits Number of qubits
     * @param threading Threading option the statevector to use
     * @param memory_model Memory model the statevector will use
     */
    explicit StateVectorLQubitManaged(
        size_t num_qubits, Threading threading = Threading::SingleThread,
        CPUMemoryModel memory_model = bestCPUMemoryModel())
        : BaseType{num_qubits, threading, memory_model},
          data_{exp2(num_qubits), ComplexPrecisionT{0.0, 0.0},
                getAllocator<ComplexPrecisionT>(this->memory_model_)} {
        data_[0] = {1, 0};
    }

    /**
     * @brief Construct a statevector from another statevector
     *
     * @tparam OtherDerived A derived type of StateVectorLQubit to use for
     * construction.
     * @param other Another statevector to construct the statevector from
     */
    template <class OtherDerived>
    explicit StateVectorLQubitManaged(
        const StateVectorLQubit<PrecisionT, OtherDerived> &other)
        : BaseType(other.getNumQubits(), other.threading(),
                   other.memoryModel()),
          data_{other.getData(), other.getData() + other.getLength(),
                getAllocator<ComplexPrecisionT>(this->memory_model_)} {}

    /**
     * @brief Construct a statevector from data pointer
     *
     * @param other_data Data pointer to construct the statevector from.
     * @param other_size Size of the data
     * @param threading Threading option the statevector to use
     * @param memory_model Memory model the statevector will use
     */
    StateVectorLQubitManaged(const ComplexPrecisionT *other_data,
                             size_t other_size,
                             Threading threading = Threading::SingleThread,
                             CPUMemoryModel memory_model = bestCPUMemoryModel())
        : BaseType(log2PerfectPower(other_size), threading, memory_model),
          data_{other_data, other_data + other_size,
                getAllocator<ComplexPrecisionT>(this->memory_model_)} {
        PL_ABORT_IF_NOT(isPerfectPowerOf2(other_size),
                        "The size of provided data must be a power of 2.");
    }

    /**
     * @brief Construct a statevector from a data vector
     *
     * @tparam Alloc Allocator type of std::vector to use for constructing
     * statevector.
     * @param other Data to construct the statevector from
     * @param threading Threading option the statevector to use
     * @param memory_model Memory model the statevector will use
     */
    template <class Alloc>
    explicit StateVectorLQubitManaged(
        const std::vector<std::complex<PrecisionT>, Alloc> &other,
        Threading threading = Threading::SingleThread,
        CPUMemoryModel memory_model = bestCPUMemoryModel())
        : StateVectorLQubitManaged(other.data(), other.size(), threading,
                                   memory_model) {}

    StateVectorLQubitManaged(const StateVectorLQubitManaged &rhs) = default;
    StateVectorLQubitManaged(StateVectorLQubitManaged &&) noexcept = default;

    StateVectorLQubitManaged &
    operator=(const StateVectorLQubitManaged &) = default;
    StateVectorLQubitManaged &
    operator=(StateVectorLQubitManaged &&) noexcept = default;

    ~StateVectorLQubitManaged() = default;

    [[nodiscard]] auto getData() -> ComplexPrecisionT * { return data_.data(); }

    [[nodiscard]] auto getData() const -> const ComplexPrecisionT * {
        return data_.data();
    }

    /**
     * @brief Get underlying data vector
     */
    [[nodiscard]] auto getDataVector()
        -> std::vector<ComplexPrecisionT, AlignedAllocator<ComplexPrecisionT>>
            & {
        return data_;
    }

    [[nodiscard]] auto getDataVector() const -> const
        std::vector<ComplexPrecisionT, AlignedAllocator<ComplexPrecisionT>> & {
        return data_;
    }

    /**
     * @brief Update data of the class to new_data
     *
     * @tparam Alloc Allocator type of std::vector to use for updating data.
     * @param new_data std::vector contains data.
     */
    template <class Alloc>
    void updateData(const std::vector<ComplexPrecisionT, Alloc> &new_data) {
        assert(data_.size() == new_data.size());
        std::copy(new_data.data(), new_data.data() + new_data.size(),
                  data_.data());
    }

    AlignedAllocator<ComplexPrecisionT> allocator() const {
        return data_.get_allocator();
    }
};
} // namespace Pennylane::LightningQubit
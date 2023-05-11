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
 * Defines the class representation for the Lightning qubit state vector.
 */

#pragma once
#include <complex>

#include "BitUtil.hpp" // log2PerfectPower, isPerfectPowerOf2
#include "Error.hpp"   // PL_ABORT
#include "StateVectorBase.hpp"

#include "BitUtil.hpp"
#include "Gates.hpp"
#include "KernelMap.hpp"
#include "KernelType.hpp"
#include "Memory.hpp"
#include "Threading.hpp"
#include "Util.hpp"

/// @cond DEV
namespace {
using Pennylane::Util::isPerfectPowerOf2;
using Pennylane::Util::log2PerfectPower;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit {
/**
 * @brief Lightning qubit state vector class.
 *
 * This class binds to a given state vector data array, and defines all
 * operations to manipulate the state vector data for quantum circuit
 * simulation. The bound data is assumed to be complex, and is required to be in
 * either 32-bit (64-bit `complex<float>`) or 64-bit (128-bit `complex<double>`)
 * floating point representation.
 *
 * @tparam PrecisionT Floating point precision of underlying state vector data.
 */
template <class PrecisionT = double>
class StateVectorLQubit
    : public StateVectorBase<PrecisionT, StateVectorLQubit<PrecisionT>> {
  public:
    using BaseType = StateVectorBase<PrecisionT, StateVectorLQubit<PrecisionT>>;
    using ComplexPrecisionT = std::complex<PrecisionT>;

  private:
    ComplexPrecisionT *data_;
    size_t length_;

  public:
    /**
     * @brief Construct state-vector from a raw data pointer.
     *
     * @param data Raw data pointer.
     * @param length The size of the data, i.e. 2^(number of qubits).
     */
    StateVectorLQubit(ComplexPrecisionT *data, size_t length)
        : BaseType{log2PerfectPower(length)}, data_{data}, length_(length) {
        // check if length is a power of 2.
        if (!isPerfectPowerOf2(length)) {
            PL_ABORT(
                "The length of the state vector must be a power of 2. But " +
                std::to_string(length) +
                " was given."); // TODO: change to std::format in C++20
        }
    }

    /**
     * @brief Get the underlying data pointer.
     *
     * @return const ComplexPrecisionT* Pointer to state vector data.
     */
    [[nodiscard]] auto getData() const -> ComplexPrecisionT * { return data_; }

    /**
     * @brief Get the underlying data pointer.
     *
     * @return ComplexPrecisionT* Pointer to state vector data.
     */
    auto getData() -> ComplexPrecisionT * { return data_; }

    /**
     * @brief Redefine statevector data pointer.
     *
     * @param data New raw data pointer.
     * @param length The size of the data, i.e. 2^(number of qubits).
     */
    void changeDataPtr(ComplexPrecisionT *data, size_t length) {
        if (!isPerfectPowerOf2(length)) {
            PL_ABORT(
                "The length of the state vector must be a power of 2. But " +
                std::to_string(length) +
                " was given."); // TODO: change to std::format in C++20
        }
        data_ = data;
        BaseType::setNumQubits(log2PerfectPower(length));
        length_ = length;
    }

    /**
     * @brief Set statevector data from another data.
     *
     * @param new_data New raw data pointer.
     * @param length The size of the data, i.e. 2^(number of qubits).
     */
    void setDataFrom(ComplexPrecisionT *new_data, size_t length) {
        if (length != this->getLength()) {
            PL_ABORT("The length of data to set must be the same as "
                     "the original data size");
        }
        std::copy(new_data, new_data + length, data_);
    }

    /**
     * @brief Get the number of data elements in the statevector array.
     *
     * @return std::size_t
     */
    [[nodiscard]] auto getLength() const -> std::size_t { return length_; }
};
} // namespace Pennylane::LightningQubit
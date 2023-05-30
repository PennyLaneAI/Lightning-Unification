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
 * Statevector simulator that binds to a given statevector data array.
 */

#pragma once
#include <complex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "BitUtil.hpp"        // log2PerfectPower, isPerfectPowerOf2
#include "CPUMemoryModel.hpp" // getMemoryModel
#include "Error.hpp"
#include "StateVectorLQubit.hpp"

#include <iostream>

/// @cond DEV
namespace {
using Pennylane::Util::getMemoryModel;
using Pennylane::Util::isPerfectPowerOf2;
using Pennylane::Util::log2PerfectPower;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit {

/**
 * @brief State-vector operations class.
 *
 * This class binds to a given statevector data array, and defines all
 * operations to manipulate the statevector data for quantum circuit simulation.
 * We define gates as methods to allow direct manipulation of the bound data, as
 * well as through a string-based function dispatch. The bound data is assumed
 * to be complex, and is required to be in either 32-bit (64-bit
 * `complex<float>`) or 64-bit (128-bit `complex<double>`) floating point
 * representation.
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data.
 */
template <class PrecisionT = double>
class StateVectorLQubitRaw final
    : public StateVectorLQubit<PrecisionT, StateVectorLQubitRaw<PrecisionT>> {
  public:
    using BaseType =
        StateVectorLQubit<PrecisionT, StateVectorLQubitRaw<PrecisionT>>;
    using ComplexPrecisionT = std::complex<PrecisionT>;

  private:
    ComplexPrecisionT *data_;
    size_t length_;

  public:
    /**
     * @brief Construct state-vector from a raw data pointer.
     *
     * Memory model is automatically deduced from a pointer.
     *
     * @param data Raw data pointer.
     * @param length The size of the data, i.e. 2^(number of qubits).
     * @param threading Threading option the statevector to use
     */
    StateVectorLQubitRaw(ComplexPrecisionT *data, size_t length,
                         Threading threading = Threading::SingleThread)
        : BaseType{log2PerfectPower(length), threading,
                   getMemoryModel(static_cast<void *>(data))},
          data_{data}, length_(length) {
        // check length is perfect power of 2
        PL_ABORT_IF_NOT(isPerfectPowerOf2(length),
                        "The size of provided data must be a power of 2.");
    }

    /**
     * @brief Get the underlying data pointer.
     *
     * @return const ComplexPrecisionT* Pointer to statevector data.
     */
    [[nodiscard]] auto getData() const -> ComplexPrecisionT * { return data_; }

    /**
     * @brief Get the underlying data pointer.
     *
     * @return ComplexPrecisionT* Pointer to statevector data.
     */
    auto getData() -> ComplexPrecisionT * { return data_; }

    /**
     * @brief Get the number of data elements in the statevector array.
     *
     * @return std::size_t
     */
    [[nodiscard]] auto getLength() const -> std::size_t { return length_; }
};
} // namespace Pennylane::LightningQubit
// Copyright 2023 Xanadu Quantum Technologies Inc.

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
 * @file StateVectorBase.hpp
 * Defines a base class for all simulators.
 * This class has methods necessary for the executor class and algorithms.
 */

#pragma once

/// @cond DEV
// Required for compilation with MSVC
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES // for C++
#endif
/// @endcond

#include <complex>

#include "Util.hpp" // exp2

namespace Pennylane {
/**
 * @brief State-vector base class.
 *
 * This class combines a data array managed by a derived class (CRTP) and
 * implementations of gate operations. The bound data is assumed to be complex,
 * and is required to be in either 32-bit (64-bit `complex<float>`) or
 * 64-bit (128-bit `complex<double>`) floating point representation.
 * As this is the base class, we do not add default template arguments.
 *
 * @tparam T Floating point precision of underlying statevector data.
 * @tparam Derived Type of a derived class
 */
template <class T, class Derived> class StateVectorBase {
  public:
    /**
     * @brief StateVector complex precision type.
     */
    using PrecisionT = T;
    using ComplexPrecisionT = std::complex<PrecisionT>;

  private:
    size_t num_qubits_{0};

  public:
    /**
     * @brief Constructor used by derived classes.
     *
     * @param num_qubits Number of qubits
     */
    explicit StateVectorBase(size_t num_qubits) : num_qubits_{num_qubits} {}

    /**
     * @brief Redefine the number of qubits in the statevector and number of
     * elements.
     *
     * @param qubits New number of qubits represented by statevector.
     */
    void setNumQubits(size_t qubits) { num_qubits_ = qubits; }

  public:
    /**
     * @brief Get the number of qubits represented by the statevector data.
     *
     * @return std::size_t
     */
    [[nodiscard]] auto getNumQubits() const -> std::size_t {
        return num_qubits_;
    }

    /**
     * @brief Get the size of the statevector
     *
     * @return The size of the statevector
     */
    [[nodiscard]] size_t getLength() const { return Util::exp2(num_qubits_); }
};

} // namespace Pennylane
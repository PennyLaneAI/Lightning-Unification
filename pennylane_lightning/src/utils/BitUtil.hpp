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
 * @file
 * Defines utility functions for Bitwise operations.
 */
#pragma once
#include <bit>
#include <cstddef>

namespace Pennylane::Util {
/**
 * @brief Faster log2 when the value is a power of 2.
 *
 * @param val Size of the state vector. Expected to be a power of 2.
 * @return size_t Log2(val), or the state vector's number of qubits.
 */
inline auto constexpr log2PerfectPower(size_t val) -> size_t {
    return static_cast<size_t>(std::countr_zero(val));
}

/**
 * @brief Verify if the value provided is a power of 2.
 *
 * @param value state vector size.
 * @return true
 * @return false
 */
inline auto constexpr isPerfectPowerOf2(size_t value) -> bool {
#if __cpp_lib_int_pow2 >= 202002L
    return std::has_single_bit(value);
#else
    return std::popcount(value) == 1;
#endif
}
} // namespace Pennylane::Util
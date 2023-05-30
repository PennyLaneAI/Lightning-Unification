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
 * Defines utility functions for Bitwise operations.
 */
#pragma once

#include "TypeTraits.hpp" // remove_complex_t

#include <cassert> // assert
#include <complex>
#include <numbers> // sqrt2_v
#include <numeric> // transform_reduce
#include <set>
#include <type_traits> // is_same_v
#include <vector>

namespace Pennylane::Util {
/**
 * @brief Compile-time scalar real times complex number.
 *
 * @tparam U Precision of real value `a`.
 * @tparam T Precision of complex value `b` and result.
 * @param a Real scalar value.
 * @param b Complex scalar value.
 * @return constexpr std::complex<T>
 */
template <class T, class U = T>
inline static constexpr auto ConstMult(U a, std::complex<T> b)
    -> std::complex<T> {
    return {a * b.real(), a * b.imag()};
}

/**
 * @brief Compile-time scalar complex times complex.
 *
 * @tparam U Precision of complex value `a`.
 * @tparam T Precision of complex value `b` and result.
 * @param a Complex scalar value.
 * @param b Complex scalar value.
 * @return constexpr std::complex<T>
 */
template <class T, class U = T>
inline static constexpr auto ConstMult(std::complex<U> a, std::complex<T> b)
    -> std::complex<T> {
    return {a.real() * b.real() - a.imag() * b.imag(),
            a.real() * b.imag() + a.imag() * b.real()};
}
template <class T, class U = T>
inline static constexpr auto ConstMultConj(std::complex<U> a, std::complex<T> b)
    -> std::complex<T> {
    return {a.real() * b.real() + a.imag() * b.imag(),
            -a.imag() * b.real() + a.real() * b.imag()};
}

/**
 * @brief Compile-time scalar complex summation.
 *
 * @tparam T Precision of complex value `a` and result.
 * @tparam U Precision of complex value `b`.
 * @param a Complex scalar value.
 * @param b Complex scalar value.
 * @return constexpr std::complex<T>
 */
template <class T, class U = T>
inline static constexpr auto ConstSum(std::complex<U> a, std::complex<T> b)
    -> std::complex<T> {
    return a + b;
}

/**
 * @brief Return complex value 1+0i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{1,0}
 */
template <class T> inline static constexpr auto ONE() -> std::complex<T> {
    return {1, 0};
}

/**
 * @brief Return complex value 0+0i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{0,0}
 */
template <class T> inline static constexpr auto ZERO() -> std::complex<T> {
    return {0, 0};
}

/**
 * @brief Return complex value 0+1i in the given precision.
 *
 * @tparam T Floating point precision type. Accepts `double` and `float`.
 * @return constexpr std::complex<T>{0,1}
 */
template <class T> inline static constexpr auto IMAG() -> std::complex<T> {
    return {0, 1};
}

/**
 * @brief Returns sqrt(2) as a compile-time constant.
 *
 * @tparam T Precision of result. `double`, `float` are accepted values.
 * @return constexpr T sqrt(2)
 */
template <class T> inline static constexpr auto SQRT2() -> T {
    return std::numbers::sqrt2_v<T>;
}

/**
 * @brief Returns inverse sqrt(2) as a compile-time constant.
 *
 * @tparam T Precision of result. `double`, `float` are accepted values.
 * @return constexpr T 1/sqrt(2)
 */
template <class T> inline static constexpr auto INVSQRT2() -> T {
    return {1 / SQRT2<T>()};
}

/**
 * @brief Calculates 2^n for some integer n > 0 using bitshifts.
 *
 * @param n the exponent
 * @return value of 2^n
 */
inline auto exp2(const size_t &n) -> size_t {
    return static_cast<size_t>(1) << n;
}

/**
 * @brief Calculates the decimal value for a qubit, assuming a big-endian
 * convention.
 *
 * @param qubitIndex the index of the qubit in the range [0, qubits)
 * @param qubits the number of qubits in the circuit
 * @return decimal value for the qubit at specified index
 */
inline auto maxDecimalForQubit(size_t qubitIndex, size_t qubits) -> size_t {
    assert(qubitIndex < qubits);
    return exp2(qubits - qubitIndex - 1);
}

/**
 * @brief Define a hash function for std::pair
 */
struct PairHash {
    /**
     * @brief A hash function for std::pair
     *
     * @tparam T The type of the first element of the pair
     * @tparam U The type of the first element of the pair
     * @param p A pair to compute hash
     */
    template <typename T, typename U>
    size_t operator()(const std::pair<T, U> &p) const {
        return std::hash<T>()(p.first) ^ std::hash<U>()(p.second);
    }
};

/**
 * @brief Iterate over all enum values (if BEGIN and END are defined).
 *
 * @tparam T enum type
 * @tparam Func function to execute
 */
template <class T, class Func> void for_each_enum(Func &&func) {
    for (auto e = T::BEGIN; e != T::END;
         e = static_cast<T>(std::underlying_type_t<T>(e) + 1)) {
        func(e);
    }
}
template <class T, class U, class Func> void for_each_enum(Func &&func) {
    for (auto e1 = T::BEGIN; e1 != T::END;
         e1 = static_cast<T>(std::underlying_type_t<T>(e1) + 1)) {
        for (auto e2 = U::BEGIN; e2 != U::END;
             e2 = static_cast<U>(std::underlying_type_t<U>(e2) + 1)) {
            func(e1, e2);
        }
    }
}

/**
 * @brief Streaming operator for vector data.
 *
 * @tparam T Vector data type.
 * @param os Output stream.
 * @param vec Vector data.
 * @return std::ostream&
 */
template <class T>
inline auto operator<<(std::ostream &os, const std::vector<T> &vec)
    -> std::ostream & {
    os << '[';
    if (!vec.empty()) {
        for (size_t i = 0; i < vec.size() - 1; i++) {
            os << vec[i] << ", ";
        }
        os << vec.back();
    }
    os << ']';
    return os;
}

/**
 * @brief Streaming operator for set data.
 *
 * @tparam T Vector data type.
 * @param os Output stream.
 * @param s Set data.
 * @return std::ostream&
 */
template <class T>
inline auto operator<<(std::ostream &os, const std::set<T> &s)
    -> std::ostream & {
    os << '{';
    for (const auto &e : s) {
        os << e << ",";
    }
    os << '}';
    return os;
}

/**
 * @brief @rst
 * Compute the squared norm of a real/complex vector :math:`\sum_k |v_k|^2`
 * @endrst
 *
 * @param data Data pointer
 * @param data_size Size of the data
 */
template <class T>
auto squaredNorm(const T *data, size_t data_size) -> remove_complex_t<T> {
    if constexpr (is_complex_v<T>) {
        // complex type
        using PrecisionT = remove_complex_t<T>;
        return std::transform_reduce(
            data, data + data_size, PrecisionT{}, std::plus<PrecisionT>(),
            static_cast<PrecisionT (*)(const std::complex<PrecisionT> &)>(
                &std::norm<PrecisionT>));
    } else {
        using PrecisionT = T;
        return std::transform_reduce(
            data, data + data_size, PrecisionT{}, std::plus<PrecisionT>(),
            static_cast<PrecisionT (*)(PrecisionT)>(std::norm));
    }
}

/**
 * @brief @rst
 * Compute the squared norm of a real/complex vector :math:`\sum_k |v_k|^2`
 * @endrst
 *
 * @param vec std::vector containing data
 */
template <class T, class Alloc>
auto squaredNorm(const std::vector<T, Alloc> &vec) -> remove_complex_t<T> {
    return squaredNorm(vec.data(), vec.size());
}

} // namespace Pennylane::Util
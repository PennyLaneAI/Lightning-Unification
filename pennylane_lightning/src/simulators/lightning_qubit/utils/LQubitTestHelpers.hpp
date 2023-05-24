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
 * Defines helper methods for PennyLane Lightning Simulator.
 */

#include "CPUMemoryModel.hpp" // getBestAllocator
#include "Constant.hpp"
#include "ConstantUtil.hpp" // array_has_elem, lookup
#include "Error.hpp"
#include "GateOperation.hpp"
#include "LinearAlgebra.hpp" // squaredNorm
#include "Macros.hpp"
#include "Memory.hpp" // AlignedAllocator
#include "TestKernels.hpp"
#include "Util.hpp"

#pragma once
#include <catch2/catch.hpp>

#include <algorithm>
#include <complex>
#include <random>
#include <string>
#include <vector>

/// @cond DEV
namespace {
using namespace Pennylane::Util;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit::Util {

template <typename T> using TestVector = std::vector<T, AlignedAllocator<T>>;

/**
 * @brief Multiplies every value in a dataset by a given complex scalar value.
 *
 * @tparam Data_t Precision of complex data type. Supports float and double
 * data.
 * @param data Data to be scaled.
 * @param scalar Scalar value.
 */
template <class Data_t, class Alloc>
void scaleVector(std::vector<std::complex<Data_t>, Alloc> &data,
                 std::complex<Data_t> scalar) {
    std::transform(
        data.begin(), data.end(), data.begin(),
        [scalar](const std::complex<Data_t> &c) { return c * scalar; });
}

/**
 * @brief Multiplies every value in a dataset by a given complex scalar value.
 *
 * @tparam Data_t Precision of complex data type. Supports float and double
 * data.
 * @param data Data to be scaled.
 * @param scalar Scalar value.
 */
template <class Data_t, class Alloc>
void scaleVector(std::vector<std::complex<Data_t>, Alloc> &data,
                 Data_t scalar) {
    std::transform(
        data.begin(), data.end(), data.begin(),
        [scalar](const std::complex<Data_t> &c) { return c * scalar; });
}

/**
 * @brief create |0>^N
 */
template <typename PrecisionT>
auto createZeroState(size_t num_qubits)
    -> TestVector<std::complex<PrecisionT>> {
    TestVector<std::complex<PrecisionT>> res(
        size_t{1U} << num_qubits, {0.0, 0.0},
        getBestAllocator<std::complex<PrecisionT>>());
    res[0] = std::complex<PrecisionT>{1.0, 0.0};
    return res;
}

/**
 * @brief create |+>^N
 */
template <typename PrecisionT>
auto createPlusState(size_t num_qubits)
    -> TestVector<std::complex<PrecisionT>> {
    TestVector<std::complex<PrecisionT>> res(
        size_t{1U} << num_qubits, {1.0, 0.0},
        getBestAllocator<std::complex<PrecisionT>>());
    for (auto &elem : res) {
        elem /= std::sqrt(1U << num_qubits);
    }
    return res;
}

/**
 * @brief create a random state
 */
template <typename PrecisionT, class RandomEngine>
auto createRandomStateVectorData(RandomEngine &re, size_t num_qubits)
    -> TestVector<std::complex<PrecisionT>> {

    TestVector<std::complex<PrecisionT>> res(
        size_t{1U} << num_qubits, {0.0, 0.0},
        getBestAllocator<std::complex<PrecisionT>>());
    std::uniform_real_distribution<PrecisionT> dist;
    for (size_t idx = 0; idx < (size_t{1U} << num_qubits); idx++) {
        res[idx] = {dist(re), dist(re)};
    }

    scaleVector(res, std::complex<PrecisionT>{1.0, 0.0} /
                         std::sqrt(squaredNorm(res.data(), res.size())));
    return res;
}

/**
 * @brief Create an arbitrary product state in X- or Z-basis.
 *
 * Example: createProductState("+01") will produce |+01> state.
 * Note that the wire index starts from the left.
 */
template <typename PrecisionT>
auto createProductState(std::string_view str)
    -> TestVector<std::complex<PrecisionT>> {
    using Pennylane::Util::INVSQRT2;
    TestVector<std::complex<PrecisionT>> st(
        getBestAllocator<std::complex<PrecisionT>>());
    st.resize(1U << str.length());

    std::vector<PrecisionT> zero{1.0, 0.0};
    std::vector<PrecisionT> one{0.0, 1.0};

    std::vector<PrecisionT> plus{INVSQRT2<PrecisionT>(),
                                 INVSQRT2<PrecisionT>()};
    std::vector<PrecisionT> minus{INVSQRT2<PrecisionT>(),
                                  -INVSQRT2<PrecisionT>()};

    for (size_t k = 0; k < (size_t{1U} << str.length()); k++) {
        PrecisionT elem = 1.0;
        for (size_t n = 0; n < str.length(); n++) {
            char c = str[n];
            const size_t wire = str.length() - 1 - n;
            switch (c) {
            case '0':
                elem *= zero[(k >> wire) & 1U];
                break;
            case '1':
                elem *= one[(k >> wire) & 1U];
                break;
            case '+':
                elem *= plus[(k >> wire) & 1U];
                break;
            case '-':
                elem *= minus[(k >> wire) & 1U];
                break;
            default:
                PL_ABORT("Unknown character in the argument.");
            }
        }
        st[k] = elem;
    }
    return st;
}

inline auto createWires(Gates::GateOperation op, size_t num_qubits)
    -> std::vector<size_t> {
    if (array_has_elem(Gates::Constant::multi_qubit_gates, op)) {
        std::vector<size_t> wires(num_qubits);
        std::iota(wires.begin(), wires.end(), 0);
        return wires;
    }
    switch (lookup(Gates::Constant::gate_wires, op)) {
    case 1:
        return {0};
    case 2:
        return {0, 1};
    case 3:
        return {0, 1, 2};
    case 4:
        return {0, 1, 2, 3};
    default:
        PL_ABORT("The number of wires for a given gate is unknown.");
    }
    return {};
}

template <class PrecisionT>
auto createParams(Gates::GateOperation op) -> std::vector<PrecisionT> {
    switch (lookup(Gates::Constant::gate_num_params, op)) {
    case 0:
        return {};
    case 1:
        return {static_cast<PrecisionT>(0.312)};
    case 3:
        return {static_cast<PrecisionT>(0.128), static_cast<PrecisionT>(-0.563),
                static_cast<PrecisionT>(1.414)};
    default:
        PL_ABORT("The number of parameters for a given gate is unknown.");
    }
    return {};
}

/**
 * @brief Fills the empty vectors with the CSR (Compressed Sparse Row) sparse
 * matrix representation for a tri-diagonal + periodic boundary conditions
 * Hamiltonian.
 *
 * @tparam fp_precision data float point precision.
 * @tparam index_type integer type used as indices of the sparse matrix.
 * @param row_map the j element encodes the total number of non-zeros above
 * row j.
 * @param entries column indices.
 * @param values  matrix non-zero elements.
 * @param numRows matrix number of rows.
 */
template <class fp_precision, class index_type>
void write_CSR_vectors(std::vector<index_type> &row_map,
                       std::vector<index_type> &entries,
                       std::vector<std::complex<fp_precision>> &values,
                       index_type numRows) {
    const std::complex<fp_precision> SC_ONE = 1.0;

    row_map.resize(numRows + 1);
    for (index_type rowIdx = 1; rowIdx < (index_type)row_map.size(); ++rowIdx) {
        row_map[rowIdx] = row_map[rowIdx - 1] + 3;
    };
    const index_type numNNZ = row_map[numRows];

    entries.resize(numNNZ);
    values.resize(numNNZ);
    for (index_type rowIdx = 0; rowIdx < numRows; ++rowIdx) {
        if (rowIdx == 0) {
            entries[0] = rowIdx;
            entries[1] = rowIdx + 1;
            entries[2] = numRows - 1;

            values[0] = SC_ONE;
            values[1] = -SC_ONE;
            values[2] = -SC_ONE;
        } else if (rowIdx == numRows - 1) {
            entries[row_map[rowIdx]] = 0;
            entries[row_map[rowIdx] + 1] = rowIdx - 1;
            entries[row_map[rowIdx] + 2] = rowIdx;

            values[row_map[rowIdx]] = -SC_ONE;
            values[row_map[rowIdx] + 1] = -SC_ONE;
            values[row_map[rowIdx] + 2] = SC_ONE;
        } else {
            entries[row_map[rowIdx]] = rowIdx - 1;
            entries[row_map[rowIdx] + 1] = rowIdx;
            entries[row_map[rowIdx] + 2] = rowIdx + 1;

            values[row_map[rowIdx]] = -SC_ONE;
            values[row_map[rowIdx] + 1] = SC_ONE;
            values[row_map[rowIdx] + 2] = -SC_ONE;
        }
    }
};

/**
 * @brief Compare std::vectors with same elements data type but different
 * allocators.
 *
 * @tparam T Element data type.
 * @tparam AllocA Allocator for the first vector.
 * @tparam AllocB Allocator for the second vector.
 * @param lhs First vector
 * @param rhs Second vector
 * @return true
 * @return false
 */
template <class T, class AllocA, class AllocB>
bool operator==(const std::vector<T, AllocA> &lhs,
                const std::vector<T, AllocB> &rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t idx = 0; idx < lhs.size(); idx++) {
        if (lhs[idx] != rhs[idx]) {
            return false;
        }
    }
    return true;
}

} // namespace Pennylane::LightningQubit::Util
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

#include "Constant.hpp"
#include "ConstantUtil.hpp" // array_has_elem, lookup
#include "Error.hpp"
#include "GateOperation.hpp"
#include "Macros.hpp"
#include "TestKernels.hpp"

#pragma once
#include <catch2/catch.hpp>

#include <algorithm>
#include <complex>
#include <string>
#include <vector>

/// @cond DEV
namespace {
using namespace Pennylane::Util;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit::Util {

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
} // namespace Pennylane::LightningQubit::Util
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

#include <cstdio>
#include <vector>

#include "KokkosSparse.hpp"
#include "MeasurementsLQubit.hpp"
#include "StateVectorLQubitManaged.hpp"
#include "StateVectorLQubitRaw.hpp"
#include "Util.hpp"

#include "TestHelpers.hpp"
#include <catch2/catch.hpp>

#if defined(_MSC_VER)
#pragma warning(disable : 4305)
#endif

/// @cond DEV
namespace {
using namespace Pennylane::Util;

using namespace Pennylane::LightningQubit;
using namespace Pennylane::LightningQubit::Measures;

}; // namespace
/// @endcond

TEMPLATE_PRODUCT_TEST_CASE("Expected Values - Sparse Hamiltonian [Kokkos]",
                           "[Measurements]",
                           (StateVectorLQubitManaged, StateVectorLQubitRaw),
                           (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

    // Defining the statevector that will be measured.
    auto statevector_data = createNonTrivialState<StateVectorT>();
    StateVectorT statevector(statevector_data.data(), statevector_data.size());

    // Initializing the measurements class.
    // This object attaches to the statevector allowing several measurements.
    Measurements<StateVectorT> Measurer(statevector);

    if constexpr (USE_KOKKOS) {
        SECTION("Testing Sparse Hamiltonian:") {
            long num_qubits = 3;
            long data_size = Pennylane::Util::exp2(num_qubits);

            std::vector<long> row_map;
            std::vector<long> entries;
            std::vector<ComplexT> values;
            write_CSR_vectors(row_map, entries, values, data_size);

            PrecisionT exp_values = Measurer.expval(
                row_map.data(), static_cast<long>(row_map.size()),
                entries.data(), values.data(),
                static_cast<long>(values.size()));
            PrecisionT exp_values_ref = 0.5930885;
            REQUIRE(exp_values == Approx(exp_values_ref).margin(1e-6));
        }

        SECTION("Testing Sparse Hamiltonian (incompatible sizes):") {
            long num_qubits = 4;
            long data_size = Pennylane::Util::exp2(num_qubits);

            std::vector<long> row_map;
            std::vector<long> entries;
            std::vector<ComplexT> values;
            write_CSR_vectors(row_map, entries, values, data_size);

            PL_CHECK_THROWS_MATCHES(
                Measurer.expval(row_map.data(),
                                static_cast<long>(row_map.size()),
                                entries.data(), values.data(),
                                static_cast<long>(values.size())),
                LightningException,
                "Statevector and Hamiltonian have incompatible sizes.");
        }
    } else {
        SECTION("Testing Sparse Hamiltonian:") {
            long num_qubits = 3;
            long data_size = Pennylane::Util::exp2(num_qubits);

            std::vector<long> row_map;
            std::vector<long> entries;
            std::vector<ComplexT> values;
            write_CSR_vectors(row_map, entries, values, data_size);

            PL_CHECK_THROWS_MATCHES(
                Measurer.expval(row_map.data(),
                                static_cast<long>(row_map.size()),
                                entries.data(), values.data(),
                                static_cast<long>(values.size())),
                LightningException,
                "Executing the product of a Sparse matrix and a vector needs "
                "Kokkos and Kokkos Kernels installation.");
        }
    }
}

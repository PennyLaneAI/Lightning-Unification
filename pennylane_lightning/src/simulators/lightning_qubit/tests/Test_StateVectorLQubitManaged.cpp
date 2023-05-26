#include "LQubitTestHelpers.hpp" // createRandomStateVectorData, TestVector
#include "LinearAlgebra.hpp"     //randomUnitary
#include "StateVectorLQubitManaged.hpp"
#include "StateVectorLQubitRaw.hpp"
#include "TestHelpers.hpp"
#include "cpu_kernels/GateImplementationsPI.hpp"

#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

/**
 * @file
 *  Tests for the StateVectorLQubitManaged class.
 */

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit;
using namespace Pennylane::Util;

using Pennylane::Util::createRandomStateVectorData;
using Pennylane::Util::TestVector;

using Pennylane::LightningQubit::Util::randomUnitary;

std::mt19937_64 re{1337};
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("StateVectorLQubitManaged::StateVectorLQubitManaged",
                   "[StateVectorLQubitManaged]", float, double) {
    using PrecisionT = TestType;

    SECTION("StateVectorLQubitManaged<TestType> {size_t}") {
        REQUIRE(std::is_constructible_v<StateVectorLQubitManaged<TestType>,
                                        size_t>);
        const size_t num_qubits = 4;
        StateVectorLQubitManaged<PrecisionT> sv(num_qubits);

        REQUIRE(sv.getNumQubits() == 4);
        REQUIRE(sv.getLength() == 16);
        REQUIRE(sv.getDataVector().size() == 16);
    }
    SECTION("StateVectorLQubitManaged<TestType> {size_t}") {
        using TestVectorT = TestVector<std::complex<PrecisionT>>;
        REQUIRE(std::is_constructible_v<StateVectorLQubitManaged<TestType>,
                                        TestVectorT>);
        const size_t num_qubits = 5;
        TestVectorT st_data =
            createRandomStateVectorData<PrecisionT>(re, num_qubits);
        StateVectorLQubitManaged<PrecisionT> sv(st_data);

        REQUIRE(sv.getNumQubits() == 5);
        REQUIRE(sv.getLength() == 32);
        REQUIRE(sv.getDataVector().size() == 32);
    }
    SECTION("StateVectorLQubitManaged<TestType> {const "
            "StateVectorLQubitRaw<TestType>&}") {
        REQUIRE(
            std::is_constructible_v<StateVectorLQubitManaged<TestType>,
                                    const StateVectorLQubitRaw<TestType> &>);
    }
    SECTION("Aligned 256bit statevector") {
        const auto memory_model = CPUMemoryModel::Aligned256;
        StateVectorLQubitManaged<PrecisionT> sv(4, Threading::SingleThread,
                                                memory_model);
        /* Even when we allocate 256 bit aligned memory it is possible that the
         * alignment happens to be 512 bit */
        REQUIRE(((getMemoryModel(sv.getDataVector().data()) ==
                  CPUMemoryModel::Aligned256) ||
                 (getMemoryModel(sv.getDataVector().data()) ==
                  CPUMemoryModel::Aligned512)));
    }

    SECTION("Aligned 512bit statevector") {
        const auto memory_model = CPUMemoryModel::Aligned512;
        StateVectorLQubitManaged<PrecisionT> sv(4, Threading::SingleThread,
                                                memory_model);
        REQUIRE((getMemoryModel(sv.getDataVector().data()) ==
                 CPUMemoryModel::Aligned512));
    }
}
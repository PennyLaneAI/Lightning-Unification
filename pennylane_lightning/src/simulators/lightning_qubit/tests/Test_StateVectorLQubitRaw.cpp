#include <complex>
#include <numeric>
#include <vector>

#include "StateVectorLQubitRaw.hpp"
#include "LQubitTestHelpers.hpp" // createRandomState
#include "TestHelpers.hpp"
#include "Error.hpp" // LightningException
// #include "Util.hpp"

#include <catch2/catch.hpp>

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit;
using namespace Pennylane::Util;

using Pennylane::LightningQubit::Util::createRandomState;
} // namespace
/// @endcond

std::mt19937_64 re{1337};

TEMPLATE_TEST_CASE("StateVectorLQubitRaw::StateVectorLQubitRaw",
                   "[StateVectorLQubitRaw]", float, double) {
    using PrecisionT = TestType;

    SECTION("StateVectorLQubitRaw<TestType> {std::complex<TestType>*, size_t}") {
        const size_t num_qubits = 4;
        auto st_data = createRandomState<PrecisionT>(re, num_qubits);
        StateVectorLQubitRaw<PrecisionT> sv(st_data.data(), st_data.size());

        REQUIRE(sv.getNumQubits() == 4);
        REQUIRE(sv.getData() == st_data.data());
        REQUIRE(sv.getLength() == 16);
    }
    SECTION("StateVectorLQubitRaw<TestType> {std::complex<TestType>*, size_t}") {
        std::vector<std::complex<TestType>> st_data(14, 0.0);
        REQUIRE_THROWS(
            StateVectorLQubitRaw<PrecisionT>(st_data.data(), st_data.size()));
    }
}

// TEMPLATE_TEST_CASE("StateVectorLQubitRaw::setData", "[StateVectorLQubitRaw]", float,
//                    double) {
//     using PrecisionT = TestType;

//     SECTION("changeDataPtr correctly update data") {
//         auto st_data = createRandomState<PrecisionT>(re, 4);
//         StateVectorLQubitRaw<PrecisionT> sv(st_data.data(), st_data.size());

//         auto st_data2 = createRandomState<PrecisionT>(re, 8);
//         sv.changeDataPtr(st_data2.data(), st_data2.size());

//         REQUIRE(sv.getNumQubits() == 8);
//         REQUIRE(sv.getData() == st_data2.data());
//         REQUIRE(sv.getLength() == (1U << 8U));
//     }

//     SECTION("changeDataPtr throws an exception when the data is incorrect") {
//         auto st_data = createRandomState<PrecisionT>(re, 4);
//         StateVectorLQubitRaw<PrecisionT> sv(st_data.data(), st_data.size());

//         std::vector<std::complex<PrecisionT>> new_data(7, PrecisionT{0.0});

//         REQUIRE_THROWS_AS(sv.changeDataPtr(new_data.data(), new_data.size()),
//                           LightningException);
//     }

//     SECTION("setDataFrom correctly update data") {
//         auto st_data1 = createRandomState<PrecisionT>(re, 4);
//         auto st_data2 = createRandomState<PrecisionT>(re, 4);
//         StateVectorLQubitRaw<PrecisionT> sv(st_data1.data(), st_data1.size());

//         sv.setDataFrom(st_data2.data(),
//                        st_data2.size()); // Should update st_data1
//         REQUIRE(st_data1 == st_data2);
//     }

//     SECTION("setDataFrom throws an exception when the data is incorrect") {
//         auto st_data1 = createRandomState<PrecisionT>(re, 4);
//         auto st_data2 = createRandomState<PrecisionT>(re, 8);
//         StateVectorLQubitRaw<PrecisionT> sv(st_data1.data(), st_data1.size());

//         REQUIRE_THROWS_AS(sv.setDataFrom(st_data2.data(), st_data2.size()),
//                           LightningException);
//     }
// }
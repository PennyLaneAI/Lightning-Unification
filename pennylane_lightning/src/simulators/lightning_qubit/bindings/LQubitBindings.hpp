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
 * @file BindingsLQubit.hpp
 * Defines LightningQubit-specific operations to export to Python, other utility
 * functions interfacing with Pybind11 and support to agnostic bindings.
 */

#pragma once
#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "DynamicDispatcher.hpp"
#include "GateOperation.hpp"

#include "StateVectorLQubitRaw.hpp"

#include "TypeList.hpp"

#include "pybind11/pybind11.h"

/// @cond DEV
namespace {
using Pennylane::LightningQubit::StateVectorLQubitRaw;
} // namespace
/// @endcond

namespace py = pybind11;

namespace Pennylane::LightningQubit {

using StateVectorBackends =
    Pennylane::Util::TypeList<StateVectorLQubitRaw<float>,
                              StateVectorLQubitRaw<double>, void>;

/**
 * @brief Register matrix.
 */
template <class StateVectorT>
void registerMatrix(
    StateVectorT &st,
    const pybind11::array_t<std::complex<typename StateVectorT::PrecisionT>,
                            pybind11::array::c_style |
                                pybind11::array::forcecast> &matrix,
    const std::vector<size_t> &wires, bool inverse = false) {
    using PrecisionT = typename StateVectorT::PrecisionT;
    st.applyMatrix(
        static_cast<const std::complex<PrecisionT> *>(matrix.request().ptr),
        wires, inverse);
}

/**
 * @brief Register StateVector class to pybind.
 *
 * @tparam StateVectorT Statevector type to register
 * @tparam Pyclass Pybind11's class object type
 *
 * @param pyclass Pybind11's class object to bind statevector
 */
template <class StateVectorT, class PyClass>
void registerGatesForStateVector(PyClass &pyclass) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision.
    using ParamT = PrecisionT;             // Parameter's data precision

    using Gates::GateOperation;
    using Pennylane::Util::for_each_enum;
    namespace Constant = Gates::Constant;

    pyclass.def("applyMatrix", &registerMatrix<StateVectorT>,
                "Apply a given matrix to wires.");

    for_each_enum<GateOperation>([&pyclass](GateOperation gate_op) {
        using Pennylane::LightningQubit::Util::lookup;
        const auto gate_name =
            std::string(lookup(Constant::gate_names, gate_op));
        const std::string doc = "Apply the " + gate_name + " gate.";
        auto func = [gate_name = gate_name](
                        StateVectorT &sv, const std::vector<size_t> &wires,
                        bool inverse, const std::vector<ParamT> &params) {
            sv.applyOperation(gate_name, wires, inverse, params);
        };
        pyclass.def(gate_name.c_str(), func, doc.c_str());
    });
}

/**
 * @brief Get a gate kernel map for a statevector
 */
template <class StateVectorT>
auto svKernelMap(const StateVectorT &sv) -> pybind11::dict {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision.
    pybind11::dict res_map;
    namespace Constant = Gates::Constant;
    using Pennylane::LightningQubit::Util::lookup;

    const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();

    auto [GateKernelMap, GeneratorKernelMap, MatrixKernelMap] =
        sv.getSupportedKernels();

    for (const auto &[gate_op, kernel] : GateKernelMap) {
        const auto key = std::string(lookup(Constant::gate_names, gate_op));
        const auto value = dispatcher.getKernelName(kernel);

        res_map[key.c_str()] = value;
    }

    for (const auto &[gen_op, kernel] : GeneratorKernelMap) {
        const auto key = std::string(lookup(Constant::generator_names, gen_op));
        const auto value = dispatcher.getKernelName(kernel);

        res_map[key.c_str()] = value;
    }

    for (const auto &[mat_op, kernel] : MatrixKernelMap) {
        const auto key = std::string(lookup(Constant::matrix_names, mat_op));
        const auto value = dispatcher.getKernelName(kernel);

        res_map[key.c_str()] = value;
    }
    return res_map;
}

/**
 * @brief Get a gate kernel map for a statevector
 */
template <class StateVectorT, class PyClass>
void registerBackendClassSpecificBindings(PyClass &pyclass) {

    registerGatesForStateVector<StateVectorT>(pyclass);

    pyclass.def("kernel_map", &svKernelMap<StateVectorT>,
                "Get internal kernels for operations");
}

/**
 * @brief Provide backend information.
 */
auto getBackendInfo() -> pybind11::dict {
    using namespace pybind11::literals;

    return pybind11::dict("NAME"_a = "lightning.qubit");
}

/**
 * @brief Register bindings for backend-specific info.
 *
 * @param m Pybind11 module.
 */
void registerBackendSpecificInfo(py::module_ &m) {
    /* Add Kokkos and Kokkos Kernels info */
    m.def("backend_info", &getBackendInfo, "Backend-specific information.");
}

} // namespace Pennylane::LightningQubit

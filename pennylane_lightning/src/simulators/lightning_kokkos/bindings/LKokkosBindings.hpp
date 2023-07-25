// Copyright 2018-2022 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "GateOperation.hpp"
#include "MeasurementsKokkos.hpp"
#include "StateVectorKokkos.hpp"
#include "TypeList.hpp"

/// @cond DEV
namespace {
using Pennylane::LightningKokkos::StateVectorKokkos;
using namespace Pennylane::LightningKokkos::Measures;
using namespace Pennylane::LightningKokkos::Algorithms;
} // namespace
/// @endcond

namespace py = pybind11;

namespace Pennylane::LightningKokkos {

using StateVectorBackends =
    Pennylane::Util::TypeList<StateVectorKokkos<float>,
                              StateVectorKokkos<double>, void>;

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
    using ComplexT = typename StateVectorT::ComplexT;
    st.applyMatrix(static_cast<const ComplexT *>(matrix.request().ptr), wires,
                   inverse);
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
        typename StateVectorT::PrecisionT; // Statevector's precision
    using ParamT = PrecisionT;             // Parameter's data precision

    using Pennylane::Gates::GateOperation;
    using Pennylane::Util::for_each_enum;
    namespace Constant = Pennylane::Gates::Constant;

    pyclass.def("applyMatrix", &registerMatrix<StateVectorT>,
                "Apply a given matrix to wires.");

    for_each_enum<GateOperation>([&pyclass](GateOperation gate_op) {
        using Pennylane::Util::lookup;
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
 * @brief Get a gate kernel map for a statevector.
 */
template <class StateVectorT, class PyClass>
void registerBackendClassSpecificBindings(PyClass &pyclass) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision
    using ComplexT = typename StateVectorT::ComplexT;
    using ParamT = PrecisionT; // Parameter's data precision
    using np_arr_c = py::array_t<std::complex<ParamT>,
                                 py::array::c_style | py::array::forcecast>;

    registerGatesForStateVector<StateVectorT>(pyclass);

    pyclass
        .def(
            "DeviceToHost",
            [](StateVectorT &device_sv, np_arr_c &host_sv) {
                py::buffer_info numpyArrayInfo = host_sv.request();
                auto *data_ptr = static_cast<ComplexT *>(numpyArrayInfo.ptr);
                if (host_sv.size()) {
                    device_sv.DeviceToHost(data_ptr, host_sv.size());
                }
            },
            "Synchronize data from the GPU device to host.")
        .def("HostToDevice",
             py::overload_cast<ComplexT *, size_t>(&StateVectorT::HostToDevice),
             "Synchronize data from the host device to GPU.")
        .def(
            "HostToDevice",
            [](StateVectorT &device_sv, const np_arr_c &host_sv) {
                const py::buffer_info numpyArrayInfo = host_sv.request();
                auto *data_ptr = static_cast<ComplexT *>(numpyArrayInfo.ptr);
                const auto length =
                    static_cast<size_t>(numpyArrayInfo.shape[0]);
                if (length) {
                    device_sv.HostToDevice(data_ptr, length);
                }
            },
            "Synchronize data from the host device to GPU.");
}

/**
 * @brief Register backend specific measurements class functionalities.
 *
 * @tparam StateVectorT
 * @tparam PyClass
 * @param pyclass Pybind11's measurements class to bind methods.
 */
template <class StateVectorT, class PyClass>
void registerBackendSpecificMeasurements(PyClass &pyclass) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision
    using ComplexT =
        typename StateVectorT::ComplexT; // Statevector's complex type
    using ParamT = PrecisionT;           // Parameter's data precision

    using np_arr_c = py::array_t<std::complex<ParamT>,
                                 py::array::c_style | py::array::forcecast>;
    using sparse_index_type = std::size_t;
    using np_arr_sparse_ind =
        py::array_t<sparse_index_type,
                    py::array::c_style | py::array::forcecast>;

    pyclass
        .def("expval",
             static_cast<PrecisionT (Measurements<StateVectorT>::*)(
                 const std::string &, const std::vector<size_t> &)>(
                 &Measurements<StateVectorT>::expval),
             "Expected value of an operation by name.")
        .def(
            "expval",
            [](Measurements<StateVectorT> &M, const np_arr_sparse_ind &row_map,
               const np_arr_sparse_ind &entries, const np_arr_c &values) {
                return M.expval(
                    static_cast<sparse_index_type *>(row_map.request().ptr),
                    static_cast<sparse_index_type>(row_map.request().size),
                    static_cast<sparse_index_type *>(entries.request().ptr),
                    static_cast<ComplexT *>(values.request().ptr),
                    static_cast<sparse_index_type>(values.request().size));
            },
            "Expected value of a sparse Hamiltonian.")
        .def("var",
             [](Measurements<StateVectorT> &M, const std::string &operation,
                const std::vector<size_t> &wires) {
                 return M.var(operation, wires);
             })
        .def("var",
             static_cast<PrecisionT (Measurements<StateVectorT>::*)(
                 const std::string &, const std::vector<size_t> &)>(
                 &Measurements<StateVectorT>::var),
             "Variance of an operation by name.");
    // pyclass.def(
    //     "var",
    //     [](Measurements<StateVectorT> &M, const np_arr_sparse_ind
    //     &row_map,
    //        const np_arr_sparse_ind &entries, const np_arr_c &values) {
    //         return M.var(
    //             static_cast<sparse_index_type *>(row_map.request().ptr),
    //             static_cast<sparse_index_type>(row_map.request().size),
    //             static_cast<sparse_index_type *>(entries.request().ptr),
    //             static_cast<ComplexT *>(
    //                 values.request().ptr),
    //             static_cast<sparse_index_type>(values.request().size));
    //     },
    //     "Variance of a sparse Hamiltonian.");
}

/**
 * @brief Register backend specific adjoint Jacobian methods.
 *
 * @tparam StateVectorT
 * @param m Pybind module
 */
template <class StateVectorT>
void registerBackendSpecificAlgorithms([[maybe_unused]] py::module_ &m) {}

/**
 * @brief Provide backend information.
 */
auto getBackendInfo() -> pybind11::dict {
    using namespace pybind11::literals;

    return pybind11::dict("NAME"_a = "lightning.kokkos", "USE_KOKKOS"_a = true);
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
} // namespace Pennylane::LightningKokkos

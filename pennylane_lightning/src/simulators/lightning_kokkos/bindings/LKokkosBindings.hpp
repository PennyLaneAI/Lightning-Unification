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

// #include <set>
// #include <tuple>
// #include <variant>
// #include <vector>

// #include "AdjointJacobianKokkos.hpp"
// #include "Error.hpp"         // LightningException
// // #include "GetConfigInfo.hpp" // Kokkos configuration info
// #include "MeasurementsKokkos.hpp"
// #include "StateVectorKokkos.hpp"

// #include "pybind11/complex.h"
// #include "pybind11/numpy.h"
// #include "pybind11/pybind11.h"
// #include "pybind11/stl.h"
// #include "pybind11/pybind11.h"

// namespace py = pybind11;

#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "GateOperation.hpp"
#include "MeasurementsKokkos.hpp"
#include "StateVectorKokkos.hpp"
#include "TypeList.hpp"

#include "pybind11/pybind11.h"

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
    st.applyMatrix(
        static_cast<const ComplexT *>(matrix.request().ptr),
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

    registerGatesForStateVector<StateVectorT>(pyclass);

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
    using ParamT = PrecisionT;             // Parameter's data precision

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
                    static_cast<ComplexT *>(
                        values.request().ptr),
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
             "Variance of an operation by name.")
        // .def(
        //     "var",
        //     [](Measurements<StateVectorT> &M, const np_arr_sparse_ind &row_map,
        //        const np_arr_sparse_ind &entries, const np_arr_c &values) {
        //         return M.var(
        //             static_cast<sparse_index_type *>(row_map.request().ptr),
        //             static_cast<sparse_index_type>(row_map.request().size),
        //             static_cast<sparse_index_type *>(entries.request().ptr),
        //             static_cast<ComplexT *>(
        //                 values.request().ptr),
        //             static_cast<sparse_index_type>(values.request().size));
        //     },
        //     "Variance of a sparse Hamiltonian.")
;
}

/**
 * @brief Register backend specific adjoint Jacobian methods.
 *
 * @tparam StateVectorT
 * @param m Pybind module
 */
template <class StateVectorT>
void registerBackendSpecificAlgorithms([[maybe_unused]] py::module_ &m) {
}

/**
 * @brief Provide backend information.
 */
auto getBackendInfo() -> pybind11::dict {
    using namespace pybind11::literals;

    return pybind11::dict("NAME"_a = "lightning.kokkos",
                          "USE_KOKKOS"_a = true);
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
} // Pennylane::LightningKokkos
/// @cond DEV

// namespace {
// using namespace Pennylane;
// using namespace Pennylane::LightningKokkos::Algorithms;
// using namespace Pennylane::LightningKokkos::Measures;
// using namespace Pennylane::LightningKokkos::Observables;
// using std::complex;
// using std::set;
// using std::string;
// using std::vector;

// namespace py = pybind11;

// /**
//  * @brief Templated class to build all required precisions for Python module.
//  *
//  * @tparam PrecisionT Precision of the statevector data.
//  * @tparam ParamT Precision of the parameter data.
//  * @param m Pybind11 module.
//  */
// template <class PrecisionT, class ParamT>
// void StateVectorKokkos_class_bindings(py::module &m) {

//     using np_arr_r =
//         py::array_t<ParamT, py::array::c_style | py::array::forcecast>;
//     using np_arr_c = py::array_t<std::complex<ParamT>,
//                                  py::array::c_style | py::array::forcecast>;
//     using StateVectorT = StateVectorKokkos<PrecisionT>;
//     using ComplexT = StateVectorT::ComplexT;

//     // Enable module name to be based on size of complex datatype
//     const std::string bitsize =
//         std::to_string(sizeof(ComplexT) * 8);
//     std::string class_name = "LightningKokkos_C" + bitsize;

//     py::class_<StateVectorT>(m, class_name.c_str())
//         .def(py::init([](std::size_t num_qubits) {
//             return new StateVectorT(num_qubits);
//         }))
//         .def(py::init([](std::size_t num_qubits,
//                          const Kokkos::InitializationSettings &kokkos_args) {
//             return new StateVectorT(num_qubits, kokkos_args);
//         }))
//         .def(py::init([](const np_arr_c &arr) {
//             py::buffer_info numpyArrayInfo = arr.request();
//             auto *data_ptr =
//                 static_cast<ComplexT *>(numpyArrayInfo.ptr);
//             return new StateVectorT(
//                 data_ptr, static_cast<std::size_t>(arr.size()));
//         }))
//         .def(py::init([](const np_arr_c &arr,
//                          const Kokkos::InitializationSettings &kokkos_args) {
//             py::buffer_info numpyArrayInfo = arr.request();
//             auto *data_ptr =
//                 static_cast<ComplexT *>(numpyArrayInfo.ptr);
//             return new StateVectorT(
//                 data_ptr, static_cast<std::size_t>(arr.size()), kokkos_args);
//         }))
//         .def(
//             "setBasisState",
//             [](StateVectorT &sv, const size_t index) {
//                 sv.setBasisState(index);
//             },
//             "Create Basis State on Device.")
//         .def(
//             "setStateVector",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &indices, const np_arr_c &state) {
//                 const auto buffer = state.request();
//                 std::vector<Kokkos::complex<ParamT>> state_kok;
//                 if (buffer.size) {
//                     const auto ptr =
//                         static_cast<const Kokkos::complex<ParamT> *>(
//                             buffer.ptr);
//                     state_kok = std::vector<Kokkos::complex<ParamT>>{
//                         ptr, ptr + buffer.size};
//                 }
//                 sv.setStateVector(indices, state_kok);
//             },
//             "Set State Vector on device with values and their corresponding "
//             "indices for the state vector on device")
//         .def(
//             "Identity",
//             []([[maybe_unused]] StateVectorT &sv,
//                [[maybe_unused]] const std::vector<std::size_t> &wires,
//                [[maybe_unused]] bool adjoint,
//                [[maybe_unused]] const std::vector<ParamT> &params) {},
//             "Apply the Identity gate.")
//         .def(
//             "PauliX",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                [[maybe_unused]] const std::vector<ParamT> &params) {
//                 sv.applyPauliX(wires, adjoint);
//             },
//             "Apply the PauliX gate.")

//         .def(
//             "PauliY",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                [[maybe_unused]] const std::vector<ParamT> &params) {
//                 sv.applyPauliY(wires, adjoint);
//             },
//             "Apply the PauliY gate.")

//         .def(
//             "PauliZ",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                [[maybe_unused]] const std::vector<ParamT> &params) {
//                 sv.applyPauliZ(wires, adjoint);
//             },
//             "Apply the PauliZ gate.")

//         .def(
//             "Hadamard",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                [[maybe_unused]] const std::vector<ParamT> &params) {
//                 sv.applyHadamard(wires, adjoint);
//             },
//             "Apply the Hadamard gate.")

//         .def(
//             "S",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                [[maybe_unused]] const std::vector<ParamT> &params) {
//                 sv.applyS(wires, adjoint);
//             },
//             "Apply the S gate.")

//         .def(
//             "T",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                [[maybe_unused]] const std::vector<ParamT> &params) {
//                 sv.applyT(wires, adjoint);
//             },
//             "Apply the T gate.")

//         .def(
//             "CNOT",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                [[maybe_unused]] const std::vector<ParamT> &params) {
//                 sv.applyCNOT(wires, adjoint);
//             },
//             "Apply the CNOT gate.")

//         .def(
//             "SWAP",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                [[maybe_unused]] const std::vector<ParamT> &params) {
//                 sv.applySWAP(wires, adjoint);
//             },
//             "Apply the SWAP gate.")

//         .def(
//             "CSWAP",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                [[maybe_unused]] const std::vector<ParamT> &params) {
//                 sv.applyCSWAP(wires, adjoint);
//             },
//             "Apply the CSWAP gate.")

//         .def(
//             "Toffoli",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                [[maybe_unused]] const std::vector<ParamT> &params) {
//                 sv.applyToffoli(wires, adjoint);
//             },
//             "Apply the Toffoli gate.")

//         .def(
//             "CY",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                [[maybe_unused]] const std::vector<ParamT> &params) {
//                 sv.applyCY(wires, adjoint);
//             },
//             "Apply the CY gate.")

//         .def(
//             "CZ",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                [[maybe_unused]] const std::vector<ParamT> &params) {
//                 sv.applyCZ(wires, adjoint);
//             },
//             "Apply the CZ gate.")

//         .def(
//             "PhaseShift",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                const std::vector<ParamT> &params) {
//                 sv.applyPhaseShift(wires, adjoint, params);
//             },
//             "Apply the PhaseShift gate.")

//         .def("apply",
//              py::overload_cast<
//                  const vector<string> &, const vector<vector<std::size_t>> &,
//                  const vector<bool> &, const vector<vector<PrecisionT>> &>(
//                  &StateVectorT::applyOperations))

//         .def("apply", py::overload_cast<const vector<string> &,
//                                         const vector<vector<std::size_t>> &,
//                                         const vector<bool> &>(
//                           &StateVectorT::applyOperations))
//         .def(
//             "apply",
//             [](StateVectorT &sv, const std::string &str,
//                const vector<size_t> &wires, bool inv,
//                [[maybe_unused]] const std::vector<std::vector<ParamT>> &params,
//                [[maybe_unused]] const np_arr_c &gate_matrix) {
//                 const auto m_buffer = gate_matrix.request();
//                 std::vector<Kokkos::complex<ParamT>> conv_matrix;
//                 if (m_buffer.size) {
//                     const auto m_ptr =
//                         static_cast<const Kokkos::complex<ParamT> *>(
//                             m_buffer.ptr);
//                     conv_matrix = std::vector<Kokkos::complex<ParamT>>{
//                         m_ptr, m_ptr + m_buffer.size};
//                 }
//                 sv.applyOperation_std(str, wires, inv, std::vector<ParamT>{},
//                                       conv_matrix);
//             },
//             "Apply operation via the gate matrix")
//         .def("applyGenerator",
//              py::overload_cast<const std::string &, const std::vector<size_t> &,
//                                bool, const vector<PrecisionT> &>(
//                  &StateVectorT::applyGenerator))
//         .def(
//             "ControlledPhaseShift",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                const std::vector<ParamT> &params) {
//                 sv.applyControlledPhaseShift(wires, adjoint, params);
//             },
//             "Apply the ControlledPhaseShift gate.")

//         .def(
//             "RX",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                const std::vector<ParamT> &params) {
//                 sv.applyRX(wires, adjoint, params);
//             },
//             "Apply the RX gate.")

//         .def(
//             "RY",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                const std::vector<ParamT> &params) {
//                 sv.applyRY(wires, adjoint, params);
//             },
//             "Apply the RY gate.")

//         .def(
//             "RZ",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                const std::vector<ParamT> &params) {
//                 sv.applyRZ(wires, adjoint, params);
//             },
//             "Apply the RZ gate.")

//         .def(
//             "Rot",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                const std::vector<ParamT> &params) {
//                 sv.applyRot(wires, adjoint, params);
//             },
//             "Apply the Rot gate.")

//         .def(
//             "CRX",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                const std::vector<ParamT> &params) {
//                 sv.applyCRX(wires, adjoint, params);
//             },
//             "Apply the CRX gate.")

//         .def(
//             "CRY",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                const std::vector<ParamT> &params) {
//                 sv.applyCRY(wires, adjoint, params);
//             },
//             "Apply the CRY gate.")

//         .def(
//             "CRZ",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                const std::vector<ParamT> &params) {
//                 sv.applyCRZ(wires, adjoint, params);
//             },
//             "Apply the CRZ gate.")

//         .def(
//             "CRot",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                const std::vector<ParamT> &params) {
//                 return sv.applyCRot(wires, adjoint, params);
//             },
//             "Apply the CRot gate.")
//         .def(
//             "IsingXX",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                const std::vector<ParamT> &params) {
//                 return sv.applyIsingXX(wires, adjoint, params);
//             },
//             "Apply the IsingXX gate.")
//         .def(
//             "IsingXY",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                const std::vector<ParamT> &params) {
//                 sv.applyIsingXY(wires, adjoint, params);
//             },
//             "Apply the IsingXY gate.")
//         .def(
//             "IsingYY",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                const std::vector<ParamT> &params) {
//                 return sv.applyIsingYY(wires, adjoint, params);
//             },
//             "Apply the IsingYY gate.")
//         .def(
//             "IsingZZ",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                const std::vector<ParamT> &params) {
//                 return sv.applyIsingZZ(wires, adjoint, params);
//             },
//             "Apply the IsingZZ gate.")
//         .def(
//             "MultiRZ",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                const std::vector<ParamT> &params) {
//                 return sv.applyMultiRZ(wires, adjoint, params);
//             },
//             "Apply the MultiRZ gate.")
//         .def(
//             "SingleExcitation",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                const std::vector<ParamT> &params) {
//                 return sv.applySingleExcitation(wires, adjoint, params);
//             },
//             "Apply the SingleExcitation gate.")
//         .def(
//             "SingleExcitationMinus",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                const std::vector<ParamT> &params) {
//                 return sv.applySingleExcitationMinus(wires, adjoint, params);
//             },
//             "Apply the SingleExcitationMinus gate.")
//         .def(
//             "SingleExcitationPlus",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                const std::vector<ParamT> &params) {
//                 return sv.applySingleExcitationPlus(wires, adjoint, params);
//             },
//             "Apply the SingleExcitationPlus gate.")
//         .def(
//             "DoubleExcitation",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                const std::vector<ParamT> &params) {
//                 return sv.applyDoubleExcitation(wires, adjoint, params);
//             },
//             "Apply the DoubleExcitation gate.")
//         .def(
//             "DoubleExcitationMinus",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                const std::vector<ParamT> &params) {
//                 return sv.applyDoubleExcitationMinus(wires, adjoint, params);
//             },
//             "Apply the DoubleExcitationMinus gate.")
//         .def(
//             "DoubleExcitationPlus",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires, bool adjoint,
//                const std::vector<ParamT> &params) {
//                 return sv.applyDoubleExcitationPlus(wires, adjoint, params);
//             },
//             "Apply the DoubleExcitationPlus gate.")
//         .def(
//             "ExpectationValue",
//             [](StateVectorT &sv, const std::string &obsName,
//                const std::vector<std::size_t> &wires,
//                [[maybe_unused]] const std::vector<ParamT> &params,
//                [[maybe_unused]] const np_arr_c &gate_matrix) {
//                 const auto m_buffer = gate_matrix.request();
//                 std::vector<Kokkos::complex<ParamT>> conv_matrix;
//                 if (m_buffer.size) {
//                     auto m_ptr =
//                         static_cast<Kokkos::complex<ParamT> *>(m_buffer.ptr);
//                     conv_matrix = std::vector<Kokkos::complex<ParamT>>{
//                         m_ptr, m_ptr + m_buffer.size};
//                 }
//                 // Return the real component only
//                 return Measurements<StateVectorT>(sv).getExpectationValue(
//                     obsName, wires, params, conv_matrix);
//             },
//             "Calculate the expectation value of the given observable.")
//         .def(
//             "ExpectationValue",
//             [](StateVectorT &sv,
//                const std::vector<std::string> &obsName,
//                const std::vector<std::size_t> &wires,
//                [[maybe_unused]] const std::vector<std::vector<ParamT>> &params,
//                [[maybe_unused]] const np_arr_c &gate_matrix) {
//                 std::string obs_concat{"#"};
//                 for (const auto &sub : obsName) {
//                     obs_concat += sub;
//                 }
//                 const auto m_buffer = gate_matrix.request();
//                 std::vector<Kokkos::complex<ParamT>> conv_matrix;
//                 if (m_buffer.size) {
//                     const auto m_ptr =
//                         static_cast<const Kokkos::complex<ParamT> *>(
//                             m_buffer.ptr);
//                     conv_matrix = std::vector<Kokkos::complex<ParamT>>{
//                         m_ptr, m_ptr + m_buffer.size};
//                 }
//                 // Return the real component only & ignore params
//                 return Measurements<StateVectorT>(sv).getExpectationValue(
//                     obs_concat, wires, std::vector<ParamT>{}, conv_matrix);
//             },
//             "Calculate the expectation value of the given observable.")
//         .def(
//             "ExpectationValue",
//             [](StateVectorT &sv,
//                const std::vector<std::size_t> &wires,
//                const np_arr_c &gate_matrix) {
//                 const auto m_buffer = gate_matrix.request();
//                 std::vector<Kokkos::complex<ParamT>> conv_matrix;
//                 if (m_buffer.size) {
//                     const auto m_ptr =
//                         static_cast<const Kokkos::complex<ParamT> *>(
//                             m_buffer.ptr);
//                     conv_matrix = std::vector<Kokkos::complex<ParamT>>{
//                         m_ptr, m_ptr + m_buffer.size};
//                 }
//                 // Return the real component only & ignore params
//                 return Measurements<StateVectorT>(sv).getExpectationValue(
//                     wires, conv_matrix);
//             },
//             "Calculate the expectation value of the given observable.")

//         .def(
//             "ExpectationValue",
//             [](StateVectorT &sv, const np_arr_c &gate_data,
//                const std::vector<std::size_t> &indices,
//                const std::vector<std::size_t> &index_ptr) {
//                 const auto m_buffer = gate_data.request();
//                 std::vector<Kokkos::complex<ParamT>> conv_data;
//                 if (m_buffer.size) {
//                     const auto m_ptr =
//                         static_cast<const Kokkos::complex<ParamT> *>(
//                             m_buffer.ptr);
//                     conv_data = std::vector<Kokkos::complex<ParamT>>{
//                         m_ptr, m_ptr + m_buffer.size};
//                 }
//                 // Return the real component only & ignore params
//                 return Measurements<StateVectorT>(sv).getExpectationValue(
//                     conv_data, indices, index_ptr);
//             },
//             "Calculate the expectation value of the given observable.")
//         .def("probs",
//              [](StateVectorT &sv,
//                 const std::vector<size_t> &wires) {
//                  auto m = Measurements<StateVectorT>(sv);
//                  if (wires.empty()) {
//                      return py::array_t<ParamT>(py::cast(m.probs()));
//                  }

//                  const bool is_sorted_wires =
//                      std::is_sorted(wires.begin(), wires.end());

//                  if (wires.size() == sv.getNumQubits()) {
//                      if (is_sorted_wires)
//                          return py::array_t<ParamT>(py::cast(m.probs()));
//                  }
//                  return py::array_t<ParamT>(py::cast(m.probs(wires)));
//              })
//         .def("GenerateSamples",
//              [](StateVectorT &sv, size_t num_wires,
//                 size_t num_shots) {
//                  auto &&result =
//                      Measurements<StateVectorT>(sv).generate_samples(num_shots);

//                  const size_t ndim = 2;
//                  const std::vector<size_t> shape{num_shots, num_wires};
//                  constexpr auto sz = sizeof(size_t);
//                  const std::vector<size_t> strides{sz * num_wires, sz};
//                  // return 2-D NumPy array
//                  return py::array(py::buffer_info(
//                      result.data(), /* data as contiguous array  */
//                      sz,            /* size of one scalar        */
//                      py::format_descriptor<size_t>::format(), /* data type */
//                      ndim,   /* number of dimensions      */
//                      shape,  /* shape of the matrix       */
//                      strides /* strides for each axis     */
//                      ));
//              })
//         .def(
//             "DeviceToHost",
//             [](StateVectorT &device_sv, np_arr_c &host_sv) {
//                 py::buffer_info numpyArrayInfo = host_sv.request();
//                 auto *data_ptr = static_cast<ComplexT *>(
//                     numpyArrayInfo.ptr);
//                 if (host_sv.size()) {
//                     device_sv.DeviceToHost(data_ptr, host_sv.size());
//                 }
//             },
//             "Synchronize data from the GPU device to host.")
//         .def("HostToDevice",
//              py::overload_cast<ComplexT *, size_t>(
//                  &StateVectorT::HostToDevice),
//              "Synchronize data from the host device to GPU.")
//         .def(
//             "HostToDevice",
//             [](StateVectorT &device_sv,
//                const np_arr_c &host_sv) {
//                 const py::buffer_info numpyArrayInfo = host_sv.request();
//                 auto *data_ptr = static_cast<ComplexT *>(
//                     numpyArrayInfo.ptr);
//                 const auto length =
//                     static_cast<size_t>(numpyArrayInfo.shape[0]);
//                 if (length) {
//                     device_sv.HostToDevice(data_ptr, length);
//                 }
//             },
//             "Synchronize data from the host device to GPU.")
//         .def("numQubits", &StateVectorT::getNumQubits)
//         .def("dataLength", &StateVectorT::getLength)
//         .def("resetKokkos", &StateVectorT::resetStateVector);

//     //***********************************************************************//
//     //                              Observable
//     //***********************************************************************//

//     class_name = "ObservableKokkos_C" + bitsize;
//     py::class_<Observable<StateVectorT>,
//                std::shared_ptr<Observable<StateVectorT>>>(
//         m, class_name.c_str(), py::module_local());

//     class_name = "NamedObsKokkos_C" + bitsize;
//     py::class_<NamedObs<StateVectorT>,
//                std::shared_ptr<NamedObs<StateVectorT>>,
//                Observable<StateVectorT>>(m, class_name.c_str(),
//                                              py::module_local())
//         .def(py::init(
//             [](const std::string &name, const std::vector<size_t> &wires) {
//                 return NamedObs<StateVectorT>(name, wires);
//             }))
//         .def("__repr__", &NamedObs<StateVectorT>::getObsName)
//         .def("get_wires", &NamedObs<StateVectorT>::getWires,
//              "Get wires of observables")
//         .def(
//             "__eq__",
//             [](const NamedObs<StateVectorT> &self,
//                py::handle other) -> bool {
//                 if (!py::isinstance<NamedObs<StateVectorT>>(other)) {
//                     return false;
//                 }
//                 auto other_cast = other.cast<NamedObs<StateVectorT>>();
//                 return self == other_cast;
//             },
//             "Compare two observables");

//     class_name = "HermitianObsKokkos_C" + bitsize;
//     py::class_<HermitianObs<StateVectorT>,
//                std::shared_ptr<HermitianObs<StateVectorT>>,
//                Observable<StateVectorT>>(m, class_name.c_str(),
//                                              py::module_local())
//         .def(py::init([](const np_arr_c &matrix,
//                          const std::vector<size_t> &wires) {
//             const auto m_buffer = matrix.request();
//             std::vector<ComplexT> conv_matrix;
//             if (m_buffer.size) {
//                 const auto m_ptr =
//                     static_cast<const ComplexT *>(m_buffer.ptr);
//                 conv_matrix = std::vector<ComplexT>{
//                     m_ptr, m_ptr + m_buffer.size};
//             }
//             return HermitianObs<StateVectorT>(conv_matrix, wires);
//         }))
//         .def("__repr__", &HermitianObs<StateVectorT>::getObsName)
//         .def("get_wires", &HermitianObs<StateVectorT>::getWires,
//              "Get wires of observables")
//         .def(
//             "__eq__",
//             [](const HermitianObs<StateVectorT> &self,
//                py::handle other) -> bool {
//                 if (!py::isinstance<HermitianObs<StateVectorT>>(other)) {
//                     return false;
//                 }
//                 auto other_cast = other.cast<HermitianObs<StateVectorT>>();
//                 return self == other_cast;
//             },
//             "Compare two observables");

//     class_name = "TensorProdObsKokkos_C" + bitsize;
//     py::class_<TensorProdObs<StateVectorT>,
//                std::shared_ptr<TensorProdObs<StateVectorT>>,
//                Observable<StateVectorT>>(m, class_name.c_str(),
//                                              py::module_local())
//         .def(py::init(
//             [](const std::vector<std::shared_ptr<Observable<StateVectorT>>>
//                    &obs) { return TensorProdObs<StateVectorT>(obs); }))
//         .def("__repr__", &TensorProdObs<StateVectorT>::getObsName)
//         .def("get_wires", &TensorProdObs<StateVectorT>::getWires,
//              "Get wires of observables")
//         .def(
//             "__eq__",
//             [](const TensorProdObs<StateVectorT> &self,
//                py::handle other) -> bool {
//                 if (!py::isinstance<TensorProdObs<StateVectorT>>(other)) {
//                     return false;
//                 }
//                 auto other_cast = other.cast<TensorProdObs<StateVectorT>>();
//                 return self == other_cast;
//             },
//             "Compare two observables");

//     class_name = "HamiltonianKokkos_C" + bitsize;
//     using ObsPtr = std::shared_ptr<Observable<StateVectorT>>;
//     py::class_<Hamiltonian<StateVectorT>,
//                std::shared_ptr<Hamiltonian<StateVectorT>>,
//                Observable<StateVectorT>>(m, class_name.c_str(),
//                                              py::module_local())
//         .def(py::init(
//             [](const np_arr_r &coeffs, const std::vector<ObsPtr> &obs) {
//                 auto buffer = coeffs.request();
//                 const auto ptr = static_cast<const ParamT *>(buffer.ptr);
//                 return Hamiltonian<StateVectorT>{
//                     std::vector(ptr, ptr + buffer.size), obs};
//             }))
//         .def("__repr__", &Hamiltonian<StateVectorT>::getObsName)
//         .def("get_wires", &Hamiltonian<StateVectorT>::getWires,
//              "Get wires of observables")
//         .def(
//             "__eq__",
//             [](const Hamiltonian<StateVectorT> &self,
//                py::handle other) -> bool {
//                 if (!py::isinstance<Hamiltonian<StateVectorT>>(other)) {
//                     return false;
//                 }
//                 auto other_cast = other.cast<Hamiltonian<StateVectorT>>();
//                 return self == other_cast;
//             },
//             "Compare two observables");

//     // class_name = "SparseHamiltonianKokkos_C" + bitsize;
//     // py::class_<SparseHamiltonian<StateVectorT>,
//     //            std::shared_ptr<SparseHamiltonian<StateVectorT>>,
//     //            Observable<StateVectorT>>(m, class_name.c_str(),
//     //                                          py::module_local())
//     //     .def(py::init([](const np_arr_c &data,
//     //                      const std::vector<std::size_t> &indices,
//     //                      const std::vector<std::size_t> &indptr,
//     //                      const std::vector<std::size_t> &wires) {
//     //         const py::buffer_info buffer_data = data.request();
//     //         const auto *data_ptr =
//     //             static_cast<ComplexT *>(buffer_data.ptr);

//     //         return SparseHamiltonian<StateVectorT>{
//     //             std::vector<ComplexT>(
//     //                 {data_ptr, data_ptr + data.size()}),
//     //             indices, indptr, wires};
//     //     }))
//     //     .def("__repr__", &SparseHamiltonian<StateVectorT>::getObsName)
//     //     .def("get_wires", &SparseHamiltonian<StateVectorT>::getWires,
//     //          "Get wires of observables")
//     //     .def(
//     //         "__eq__",
//     //         [](const SparseHamiltonian<StateVectorT> &self,
//     //            py::handle other) -> bool {
//     //             if (!py::isinstance<SparseHamiltonian<StateVectorT>>(
//     //                     other)) {
//     //                 return false;
//     //             }
//     //             auto other_cast =
//     //                 other.cast<SparseHamiltonian<StateVectorT>>();
//     //             return self == other_cast;
//     //         },
//     //         "Compare two observables");

//     //***********************************************************************//
//     //                              Operations
//     //***********************************************************************//
//     class_name = "OpsStructKokkos_C" + bitsize;
//     py::class_<OpsData<StateVectorT>>(m, class_name.c_str(), py::module_local())
//         .def(py::init<
//              const std::vector<std::string> &,
//              const std::vector<std::vector<ParamT>> &,
//              const std::vector<std::vector<size_t>> &,
//              const std::vector<bool> &,
//              const std::vector<std::vector<ComplexT>> &>())
//         .def("__repr__", [](const OpsData<StateVectorT> &ops) {
//             using namespace Pennylane::LightningKokkos::Util;
//             std::ostringstream ops_stream;
//             for (size_t op = 0; op < ops.getSize(); op++) {
//                 ops_stream << "{'name': " << ops.getOpsName()[op];
//                 ops_stream << ", 'params': " << ops.getOpsParams()[op];
//                 ops_stream << ", 'inv': " << ops.getOpsInverses()[op];
//                 ops_stream << "}";
//                 if (op < ops.getSize() - 1) {
//                     ops_stream << ",";
//                 }
//             }
//             return "Operations: [" + ops_stream.str() + "]";
//         });

//     //***********************************************************************//
//     //                              Adj Jac
//     //***********************************************************************//

//     class_name = "AdjointJacobianKokkos_C" + bitsize;
//     py::class_<AdjointJacobian<StateVectorT>>(m, class_name.c_str(),
//                                                   py::module_local())
//         .def(py::init<>())
//         .def("create_ops_list",
//              [](AdjointJacobian<StateVectorT> &adj,
//                 const std::vector<std::string> &ops_name,
//                 const std::vector<np_arr_r> &ops_params,
//                 const std::vector<std::vector<size_t>> &ops_wires,
//                 const std::vector<bool> &ops_inverses,
//                 const std::vector<np_arr_c> &ops_matrices) {
//                  std::vector<std::vector<PrecisionT>> conv_params(
//                      ops_params.size());
//                  std::vector<std::vector<ComplexT>>
//                      conv_matrices(ops_matrices.size());
//                  static_cast<void>(adj);
//                  for (size_t op = 0; op < ops_name.size(); op++) {
//                      const auto p_buffer = ops_params[op].request();
//                      const auto m_buffer = ops_matrices[op].request();
//                      if (p_buffer.size) {
//                          const auto *const p_ptr =
//                              static_cast<const ParamT *>(p_buffer.ptr);
//                          conv_params[op] =
//                              std::vector<ParamT>{p_ptr, p_ptr + p_buffer.size};
//                      }

//                      if (m_buffer.size) {
//                          const auto m_ptr =
//                              static_cast<const ComplexT *>(
//                                  m_buffer.ptr);
//                          conv_matrices[op] = std::vector<ComplexT>{
//                              m_ptr, m_ptr + m_buffer.size};
//                      }
//                  }

//                  return OpsData<StateVectorT>{ops_name, conv_params, ops_wires,
//                                             ops_inverses, conv_matrices};
//              })
// //         .def("adjoint_jacobian",
// //              &AdjointJacobian<StateVectorT>::adjointJacobian)
//         .def("adjoint_jacobian",
//              [](AdjointJacobian<StateVectorT> &adj,
//                 const StateVectorT &sv,
//                 const std::vector<std::shared_ptr<Observable<StateVectorT>>>
//                     &observables,
//                 const OpsData<StateVectorT> &operations,
//                 const std::vector<size_t> &trainableParams) {
//                  std::vector<std::vector<PrecisionT>> jac(
//                      observables.size(),
//                      std::vector<PrecisionT>(trainableParams.size(), 0));
//                  adj.adjointJacobian(sv, jac, observables, operations,
//                                      trainableParams, false);
//                  return py::array_t<ParamT>(py::cast(jac));
//              })
// ;
// }

// /**
//  * @brief Streaming operator for Kokkos::InitializationSettings objects.
//  *
//  * @param os Output stream.
//  * @param args Kokkos::InitializationSettings object.
//  * @return std::ostream&
//  */
// inline auto operator<<(std::ostream &os,
//                        const Kokkos::InitializationSettings &args)
//     -> std::ostream & {
//     os << "InitializationSettings:\n";
//     os << "num_threads = " << args.get_num_threads() << '\n';
//     os << "device_id = " << args.get_device_id() << '\n';
//     os << "map_device_id_by = " << args.get_map_device_id_by() << '\n';
//     os << "disable_warnings = " << args.get_disable_warnings() << '\n';
//     os << "print_configuration = " << args.get_print_configuration() << '\n';
//     os << "tune_internals = " << args.get_tune_internals() << '\n';
//     os << "tools_libs = " << args.get_tools_libs() << '\n';
//     os << "tools_help = " << args.get_tools_help() << '\n';
//     os << "tools_args = " << args.get_tools_args();
//     return os;
// }

// // Necessary to avoid mangled names when manually building module
// // due to CUDA & LTO incompatibility issues.
// extern "C" {
// /**
//  * @brief Add C++ classes, methods and functions to Python module.
//  */
// PYBIND11_MODULE(lightning_kokkos_qubit_ops, // NOLINT: No control over
//                                             // Pybind internals
//                 m) {
//     // Suppress doxygen autogenerated signatures

//     py::options options;
//     options.disable_function_signatures();
//     py::register_exception<LightningException>(m, "PLException");

//     StateVectorKokkos_class_bindings<float, float>(m);
//     StateVectorKokkos_class_bindings<double, double>(m);

//     m.def("kokkos_start", []() { Kokkos::initialize(); });
//     m.def("kokkos_end", []() { Kokkos::finalize(); });
//     m.def("kokkos_config_info", &getConfig, "Kokkos configurations query.");
//     m.def(
//         "print_configuration",
//         []() {
//             std::ostringstream buffer;
//             Kokkos::print_configuration(buffer, true);
//             return buffer.str();
//         },
//         "Kokkos configurations query.");

//     py::class_<Kokkos::InitializationSettings>(m, "InitializationSettings")
//         .def(py::init([]() {
//             return Kokkos::InitializationSettings()
//                 .set_num_threads(0)
//                 .set_device_id(0)
//                 .set_map_device_id_by("")
//                 .set_disable_warnings(0)
//                 .set_print_configuration(0)
//                 .set_tune_internals(0)
//                 .set_tools_libs("")
//                 .set_tools_help(0)
//                 .set_tools_args("");
//         }))
//         .def("get_num_threads",
//              &Kokkos::InitializationSettings::get_num_threads,
//              "Number of threads to use with the host parallel backend. Must be "
//              "greater than zero.")
//         .def("get_device_id", &Kokkos::InitializationSettings::get_device_id,
//              "Device to use with the device parallel backend. Valid IDs are "
//              "zero to number of GPU(s) available for execution minus one.")
//         .def(
//             "get_map_device_id_by",
//             &Kokkos::InitializationSettings::get_map_device_id_by,
//             "Strategy to select a device automatically from the GPUs available "
//             "for execution. Must be either mpi_rank"
//             "for round-robin assignment based on the local MPI rank or random.")
//         .def("get_disable_warnings",
//              &Kokkos::InitializationSettings::get_disable_warnings,
//              "Whether to disable warning messages.")
//         .def("get_print_configuration",
//              &Kokkos::InitializationSettings::get_print_configuration,
//              "Whether to print the configuration after initialization.")
//         .def("get_tune_internals",
//              &Kokkos::InitializationSettings::get_tune_internals,
//              "Whether to allow autotuning internals instead of using "
//              "heuristics.")
//         .def("get_tools_libs", &Kokkos::InitializationSettings::get_tools_libs,
//              "Which tool dynamic library to load. Must either be the full path "
//              "to library or the name of library if the path is present in the "
//              "runtime library search path (e.g. LD_LIBRARY_PATH)")
//         .def("get_tools_help", &Kokkos::InitializationSettings::get_tools_help,
//              "Query the loaded tool for its command-line options support.")
//         .def("get_tools_args", &Kokkos::InitializationSettings::get_tools_args,
//              "Options to pass to the loaded tool as command-line arguments.")
//         .def("has_num_threads",
//              &Kokkos::InitializationSettings::has_num_threads,
//              "Number of threads to use with the host parallel backend. Must be "
//              "greater than zero.")
//         .def("has_device_id", &Kokkos::InitializationSettings::has_device_id,
//              "Device to use with the device parallel backend. Valid IDs are "
//              "zero "
//              "to number of GPU(s) available for execution minus one.")
//         .def(
//             "has_map_device_id_by",
//             &Kokkos::InitializationSettings::has_map_device_id_by,
//             "Strategy to select a device automatically from the GPUs available "
//             "for execution. Must be either mpi_rank"
//             "for round-robin assignment based on the local MPI rank or random.")
//         .def("has_disable_warnings",
//              &Kokkos::InitializationSettings::has_disable_warnings,
//              "Whether to disable warning messages.")
//         .def("has_print_configuration",
//              &Kokkos::InitializationSettings::has_print_configuration,
//              "Whether to print the configuration after initialization.")
//         .def("has_tune_internals",
//              &Kokkos::InitializationSettings::has_tune_internals,
//              "Whether to allow autotuning internals instead of using "
//              "heuristics.")
//         .def("has_tools_libs", &Kokkos::InitializationSettings::has_tools_libs,
//              "Which tool dynamic library to load. Must either be the full path "
//              "to "
//              "library or the name of library if the path is present in the "
//              "runtime library search path (e.g. LD_LIBRARY_PATH)")
//         .def("has_tools_help", &Kokkos::InitializationSettings::has_tools_help,
//              "Query the loaded tool for its command-line options support.")
//         .def("has_tools_args", &Kokkos::InitializationSettings::has_tools_args,
//              "Options to pass to the loaded tool as command-line arguments.")
//         .def("set_num_threads",
//              &Kokkos::InitializationSettings::set_num_threads,
//              "Number of threads to use with the host parallel backend. Must be "
//              "greater than zero.")
//         .def("set_device_id", &Kokkos::InitializationSettings::set_device_id,
//              "Device to use with the device parallel backend. Valid IDs are "
//              "zero to number of GPU(s) available for execution minus one.")
//         .def(
//             "set_map_device_id_by",
//             &Kokkos::InitializationSettings::set_map_device_id_by,
//             "Strategy to select a device automatically from the GPUs available "
//             "for execution. Must be either mpi_rank"
//             "for round-robin assignment based on the local MPI rank or random.")
//         .def("set_disable_warnings",
//              &Kokkos::InitializationSettings::set_disable_warnings,
//              "Whether to disable warning messages.")
//         .def("set_print_configuration",
//              &Kokkos::InitializationSettings::set_print_configuration,
//              "Whether to print the configuration after initialization.")
//         .def("set_tune_internals",
//              &Kokkos::InitializationSettings::set_tune_internals,
//              "Whether to allow autotuning internals instead of using "
//              "heuristics.")
//         .def("set_tools_libs", &Kokkos::InitializationSettings::set_tools_libs,
//              "Which tool dynamic library to load. Must either be the full path "
//              "to library or the name of library if the path is present in the "
//              "runtime library search path (e.g. LD_LIBRARY_PATH)")
//         .def("set_tools_help", &Kokkos::InitializationSettings::set_tools_help,
//              "Query the loaded tool for its command-line options support.")
//         .def("set_tools_args", &Kokkos::InitializationSettings::set_tools_args,
//              "Options to pass to the loaded tool as command-line arguments.")
//         .def("__repr__", [](const Kokkos::InitializationSettings &args) {
//             using namespace Pennylane::LightningKokkos::Util;
//             std::ostringstream args_stream;
//             args_stream << args;
//             return args_stream.str();
//         })
// ;
// }
// }

// } // namespace
//  /// @endcond

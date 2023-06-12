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
 * @file Bindings.hpp
 * Defines device-agnostic operations to export to Python and other utility
 * functions interfacing with Pybind11.
 */

#pragma once
#include "CPUMemoryModel.hpp" // CPUMemoryModel, getMemoryModel, bestCPUMemoryModel, getAlignment
#include "Macros.hpp" // CPUArch
#include "Memory.hpp" // alignedAlloc
#include "Observables.hpp"
#include "Util.hpp" // for_each_enum

#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include <cassert>
#include <iostream>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#ifdef _ENABLE_PLQUBIT
#include "LQubitBindings.hpp" // StateVectorBackends, registerBackendClassSpecificBindings
#include "ObservablesLQubit.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit;
using namespace Pennylane::LightningQubit::Observables;
} // namespace
/// @endcond
#else
static_assert(false, "Backend not found.");
#endif

/// @cond DEV
namespace {
using Pennylane::Util::bestCPUMemoryModel;
using Pennylane::Util::CPUMemoryModel;
using Pennylane::Util::getMemoryModel;
} // namespace
/// @endcond

namespace py = pybind11;

namespace Pennylane {

/**
 * @brief Create a State Vector From a 1D Numpy Data object.
 *
 * @tparam StateVectorT
 * @param numpyArray inout data
 * @return StateVectorT
 */
template <class StateVectorT>
auto createStateVectorFromNumpyData(
    const pybind11::array_t<std::complex<typename StateVectorT::PrecisionT>>
        &numpyArray) -> StateVectorT {
    using PrecisionT = typename StateVectorT::PrecisionT;
    pybind11::buffer_info numpyArrayInfo = numpyArray.request();
    if (numpyArrayInfo.ndim != 1) {
        throw std::invalid_argument(
            "NumPy array must be a 1-dimensional array");
    }
    if (numpyArrayInfo.itemsize != sizeof(std::complex<PrecisionT>)) {
        throw std::invalid_argument(
            "NumPy array must be of type np.complex64 or np.complex128");
    }
    auto *data_ptr =
        static_cast<std::complex<PrecisionT> *>(numpyArrayInfo.ptr);
    return StateVectorT(
        {data_ptr, static_cast<size_t>(numpyArrayInfo.shape[0])});
}

/**
 * @brief Get memory alignment of a given numpy array.
 *
 * @param numpyArray Pybind11's numpy array type.
 * @return CPUMemoryModel Memory model describing alignment
 */
auto getNumpyArrayAlignment(const pybind11::array &numpyArray)
    -> CPUMemoryModel {
    return getMemoryModel(numpyArray.request().ptr);
}

/**
 * @brief Create an aligned numpy array for a given type, memory model and array
 * size.
 *
 * @tparam T Datatype of numpy array to create
 * @param memory_model Memory model to use
 * @param size Size of the array to create
 * @return Numpy array
 */
template <typename T>
auto alignedNumpyArray(CPUMemoryModel memory_model, size_t size)
    -> pybind11::array {
    using Pennylane::Util::alignedAlloc;
    if (getAlignment<T>(memory_model) > alignof(std::max_align_t)) {
        void *ptr =
            alignedAlloc(getAlignment<T>(memory_model), sizeof(T) * size);
        auto capsule = pybind11::capsule(ptr, &Util::alignedFree);
        return pybind11::array{
            pybind11::dtype::of<T>(), {size}, {sizeof(T)}, ptr, capsule};
    }
    void *ptr = static_cast<void *>(new T[size]);
    auto capsule =
        pybind11::capsule(ptr, [](void *p) { delete static_cast<T *>(p); });
    return pybind11::array{
        pybind11::dtype::of<T>(), {size}, {sizeof(T)}, ptr, capsule};
}
/**
 * @brief Create a numpy array whose underlying data is allocated by
 * lightning.
 *
 * See https://github.com/pybind/pybind11/issues/1042#issuecomment-325941022
 * for capsule usage.
 *
 * @param size Size of the array to create
 * @param dt Pybind11's datatype object
 */
auto allocateAlignedArray(size_t size, const pybind11::dtype &dt)
    -> pybind11::array {
    auto memory_model = bestCPUMemoryModel();

    if (dt.is(pybind11::dtype::of<float>())) {
        return alignedNumpyArray<float>(memory_model, size);
    }
    if (dt.is(pybind11::dtype::of<double>())) {
        return alignedNumpyArray<double>(memory_model, size);
    }
    if (dt.is(pybind11::dtype::of<std::complex<float>>())) {
        return alignedNumpyArray<std::complex<float>>(memory_model, size);
    }
    if (dt.is(pybind11::dtype::of<std::complex<double>>())) {
        return alignedNumpyArray<std::complex<double>>(memory_model, size);
    }
    throw pybind11::type_error("Unsupported datatype.");
}

/**
 * @brief Register functionality for numpy array memory alignment.
 *
 * @param m Pybind module
 */
void registerArrayAlignmentBindings(py::module_ &m) {
    /* Add CPUMemoryModel enum class */
    pybind11::enum_<CPUMemoryModel>(m, "CPUMemoryModel")
        .value("Unaligned", CPUMemoryModel::Unaligned)
        .value("Aligned256", CPUMemoryModel::Aligned256)
        .value("Aligned512", CPUMemoryModel::Aligned512);

    /* Add array alignment functionality */
    m.def("get_alignment", &getNumpyArrayAlignment,
          "Get alignment of an underlying data for a numpy array.");
    m.def("allocate_aligned_array", &allocateAlignedArray,
          "Get numpy array whose underlying data is aligned.");
    m.def("best_alignment", &bestCPUMemoryModel,
          "Best memory alignment. for the simulator.");
}

/**
 * @brief Register Observable Classes
 *
 * @tparam StateVectorT
 * @param m Pybind module
 */
template <class StateVectorT> void registerObservables(py::module_ &m) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision.
    using ParamT = PrecisionT;             // Parameter's data precision

    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);

    using np_arr_c = py::array_t<std::complex<ParamT>, py::array::c_style>;
    using np_arr_r = py::array_t<ParamT, py::array::c_style>;

    std::string class_name;

    class_name = "ObservableC" + bitsize;
    py::class_<Observable<StateVectorT>,
               std::shared_ptr<Observable<StateVectorT>>>(m, class_name.c_str(),
                                                          py::module_local());

    class_name = "NamedObsC" + bitsize;
    py::class_<NamedObs<StateVectorT>, std::shared_ptr<NamedObs<StateVectorT>>,
               Observable<StateVectorT>>(m, class_name.c_str(),
                                         py::module_local())
        .def(py::init(
            [](const std::string &name, const std::vector<size_t> &wires) {
                return NamedObs<StateVectorT>(name, wires);
            }))
        .def("__repr__", &NamedObs<StateVectorT>::getObsName)
        .def("get_wires", &NamedObs<StateVectorT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const NamedObs<StateVectorT> &self, py::handle other) -> bool {
                if (!py::isinstance<NamedObs<StateVectorT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<NamedObs<StateVectorT>>();
                return self == other_cast;
            },
            "Compare two observables");

    class_name = "HermitianObsC" + bitsize;
    py::class_<HermitianObs<StateVectorT>,
               std::shared_ptr<HermitianObs<StateVectorT>>,
               Observable<StateVectorT>>(m, class_name.c_str(),
                                         py::module_local())
        .def(py::init([](const np_arr_c &matrix,
                         const std::vector<size_t> &wires) {
            auto buffer = matrix.request();
            const auto *ptr =
                static_cast<std::complex<PrecisionT> *>(buffer.ptr);
            return HermitianObs<StateVectorT>(
                std::vector<std::complex<PrecisionT>>(ptr, ptr + buffer.size),
                wires);
        }))
        .def("__repr__", &HermitianObs<StateVectorT>::getObsName)
        .def("get_wires", &HermitianObs<StateVectorT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const HermitianObs<StateVectorT> &self,
               py::handle other) -> bool {
                if (!py::isinstance<HermitianObs<StateVectorT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<HermitianObs<StateVectorT>>();
                return self == other_cast;
            },
            "Compare two observables");

    class_name = "TensorProdObsC" + bitsize;
    py::class_<TensorProdObs<StateVectorT>,
               std::shared_ptr<TensorProdObs<StateVectorT>>,
               Observable<StateVectorT>>(m, class_name.c_str(),
                                         py::module_local())
        .def(py::init(
            [](const std::vector<std::shared_ptr<Observable<StateVectorT>>>
                   &obs) { return TensorProdObs<StateVectorT>(obs); }))
        .def("__repr__", &TensorProdObs<StateVectorT>::getObsName)
        .def("get_wires", &TensorProdObs<StateVectorT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const TensorProdObs<StateVectorT> &self,
               py::handle other) -> bool {
                if (!py::isinstance<TensorProdObs<StateVectorT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<TensorProdObs<StateVectorT>>();
                return self == other_cast;
            },
            "Compare two observables");

    class_name = "HamiltonianC" + bitsize;
    using ObsPtr = std::shared_ptr<Observable<StateVectorT>>;
    py::class_<Hamiltonian<StateVectorT>,
               std::shared_ptr<Hamiltonian<StateVectorT>>,
               Observable<StateVectorT>>(m, class_name.c_str(),
                                         py::module_local())
        .def(py::init(
            [](const np_arr_r &coeffs, const std::vector<ObsPtr> &obs) {
                auto buffer = coeffs.request();
                const auto ptr = static_cast<const ParamT *>(buffer.ptr);
                return Hamiltonian<StateVectorT>{
                    std::vector(ptr, ptr + buffer.size), obs};
            }))
        .def("__repr__", &Hamiltonian<StateVectorT>::getObsName)
        .def("get_wires", &Hamiltonian<StateVectorT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const Hamiltonian<StateVectorT> &self,
               py::handle other) -> bool {
                if (!py::isinstance<Hamiltonian<StateVectorT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<Hamiltonian<StateVectorT>>();
                return self == other_cast;
            },
            "Compare two observables");
}

/**
 * @brief Return basic information of the compiled binary.
 */
auto getCompileInfo() -> pybind11::dict {
    using namespace Pennylane::Util;
    using namespace pybind11::literals;

    const std::string_view cpu_arch_str = [] {
        switch (cpu_arch) {
        case CPUArch::X86_64:
            return "x86_64";
        case CPUArch::PPC64:
            return "PPC64";
        case CPUArch::ARM:
            return "ARM";
        default:
            return "Unknown";
        }
    }();

    const std::string_view compiler_name_str = [] {
        switch (compiler) {
        case Compiler::GCC:
            return "GCC";
        case Compiler::Clang:
            return "Clang";
        case Compiler::MSVC:
            return "MSVC";
        case Compiler::NVCC:
            return "NVCC";
        case Compiler::NVHPC:
            return "NVHPC";
        default:
            return "Unknown";
        }
    }();

    const auto compiler_version_str = getCompilerVersion<compiler>();

    return pybind11::dict("cpu.arch"_a = cpu_arch_str,
                          "compiler.name"_a = compiler_name_str,
                          "compiler.version"_a = compiler_version_str,
                          "AVX2"_a = use_avx2, "AVX512F"_a = use_avx512f);
}

/**
 * @brief Return basic information of runtime environment
 */
auto getRuntimeInfo() -> pybind11::dict {
    using Pennylane::Util::RuntimeInfo;
    using namespace pybind11::literals;

    return pybind11::dict("AVX"_a = RuntimeInfo::AVX(),
                          "AVX2"_a = RuntimeInfo::AVX2(),
                          "AVX512F"_a = RuntimeInfo::AVX512F());
}

/**
 * @brief Register bindings for general info.
 *
 * @param m Pybind11 module.
 */
void registerInfo(py::module_ &m) {
    /* Add compile info */
    m.def("compile_info", &getCompileInfo, "Compiled binary information.");

    /* Add runtime info */
    m.def("runtime_info", &getRuntimeInfo, "Runtime information.");
}

/**
 * @brief Templated class to build lightning class bindings.
 *
 * @tparam StateVectorT State vector type
 * @param m Pybind11 module.
 */
template <class StateVectorT> void lightningClassBindings(py::module_ &m) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision.
    // Enable module name to be based on size of complex datatype
    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);

    //***********************************************************************//
    //                              StateVector
    //***********************************************************************//

    std::string class_name = "StateVectorC" + bitsize;
    auto pyclass =
        py::class_<StateVectorT>(m, class_name.c_str(), py::module_local());
    pyclass.def(py::init(&createStateVectorFromNumpyData<StateVectorT>));

    registerBackendClassSpecificBindings<StateVectorT>(pyclass);

    //***********************************************************************//
    //                              Observables
    //***********************************************************************//

    /* Observables submodule */
    py::module_ obs_submodule =
        m.def_submodule("observables", "Submodule for observables.");
    registerObservables<StateVectorT>(obs_submodule);
}

template <typename TypeList>
void registerLightningClassBindings(py::module_ &m) {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        lightningClassBindings<StateVectorT>(m);
        registerLightningClassBindings<typename TypeList::Next>(m);
    }
}
} // namespace Pennylane
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
 * @file Bindings.cpp
 * Export C++ functions to Python using Pybind.
 */
#include "Bindings.hpp"

#include "pybind11/pybind11.h"

/// @cond DEV
namespace {
using namespace Pennylane;
} // namespace
/// @endcond

/**
 * @brief Add C++ classes, methods and functions to Python module.
 */
PYBIND11_MODULE(
    pennylane_lightning_ops, // NOLINT: No control over Pybind internals
    m) {
    // Suppress doxygen autogenerated signatures

    pybind11::options options;
    options.disable_function_signatures();

    // Register functionality for numpy array memory alignment:
    registerArrayAlignmentBindings(m);

    // Register bindings for general info:
    registerInfo(m);

    // Register bindings for backend-specific info:
    registerBackendSpecificInfo(m);

    registerLightningClassBindings<StateVectorBackends>(m);
}
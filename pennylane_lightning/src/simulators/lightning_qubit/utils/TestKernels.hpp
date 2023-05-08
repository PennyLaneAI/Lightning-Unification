#pragma once
/**
 * @file
 * We define default test kernels.
 */
#include "Macros.hpp"
#include "TypeList.hpp"

#include "cpu_kernels/GateImplementationsLM.hpp"
#include "cpu_kernels/GateImplementationsPI.hpp"

using TestKernels = Pennylane::Lightning_Qubit::Util::TypeList<
    Pennylane::Lightning_Qubit::Gates::GateImplementationsLM,
    Pennylane::Lightning_Qubit::Gates::GateImplementationsPI, void>;

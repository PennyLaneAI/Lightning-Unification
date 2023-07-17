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

#include "AdjointDiffKokkos.hpp"
#include "JacobianData.hpp"

// using namespace Pennylane;
using namespace Pennylane::LightningKokkos::Algorithms;
using Pennylane::LightningKokkos::StateVectorKokkos;

// explicit instantiation
template class Pennylane::Algorithms::OpsData<StateVectorKokkos<float>>;
template class Pennylane::Algorithms::OpsData<StateVectorKokkos<double>>;

template class Pennylane::Algorithms::JacobianData<StateVectorKokkos<float>>;
template class Pennylane::Algorithms::JacobianData<StateVectorKokkos<double>>;

template class AdjointJacobian<StateVectorKokkos<float>>;
template class AdjointJacobian<StateVectorKokkos<double>>;

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

#include "ObservablesKokkos.hpp"
#include "StateVectorKokkos.hpp"

using namespace Pennylane::Lightning_Kokkos;

template class Observables::NamedObs<StateVectorKokkos<float>>;
template class Observables::NamedObs<StateVectorKokkos<double>>;

template class Observables::HermitianObs<StateVectorKokkos<float>>;
template class Observables::HermitianObs<StateVectorKokkos<double>>;

template class Observables::TensorProdObs<StateVectorKokkos<float>>;
template class Observables::TensorProdObs<StateVectorKokkos<double>>;

template class Observables::Hamiltonian<StateVectorKokkos<float>>;
template class Observables::Hamiltonian<StateVectorKokkos<double>>;
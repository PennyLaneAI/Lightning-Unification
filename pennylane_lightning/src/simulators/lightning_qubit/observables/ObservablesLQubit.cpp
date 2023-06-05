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

#include "ObservablesLQubit.hpp"
#include "StateVectorLQubitManaged.hpp"
#include "StateVectorLQubitRaw.hpp"

using namespace Pennylane::LightningQubit;

template class Observables::NamedObs<StateVectorLQubitRaw<float>, float>;
template class Observables::NamedObs<StateVectorLQubitRaw<double>, double>;

template class Observables::NamedObs<StateVectorLQubitManaged<float>, float>;
template class Observables::NamedObs<StateVectorLQubitManaged<double>, double>;

template class Observables::HermitianObs<StateVectorLQubitRaw<float>, float>;
template class Observables::HermitianObs<StateVectorLQubitRaw<double>, double>;

template class Observables::HermitianObs<StateVectorLQubitManaged<float>,
                                         float>;
template class Observables::HermitianObs<StateVectorLQubitManaged<double>,
                                         double>;

template class Observables::TensorProdObs<StateVectorLQubitRaw<float>, float>;
template class Observables::TensorProdObs<StateVectorLQubitRaw<double>, double>;

template class Observables::TensorProdObs<StateVectorLQubitManaged<float>,
                                          float>;
template class Observables::TensorProdObs<StateVectorLQubitManaged<double>,
                                          double>;

template class Observables::Hamiltonian<StateVectorLQubitRaw<float>, float>;
template class Observables::Hamiltonian<StateVectorLQubitRaw<double>, double>;

template class Observables::Hamiltonian<StateVectorLQubitManaged<float>, float>;
template class Observables::Hamiltonian<StateVectorLQubitManaged<double>,
                                        double>;

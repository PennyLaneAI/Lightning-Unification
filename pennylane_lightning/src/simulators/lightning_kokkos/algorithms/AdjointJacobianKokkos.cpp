#include "AdjointJacobianKokkos.hpp"
#include "StateVectorKokkos.hpp"

using namespace Pennylane::LightningKokkos;

// explicit instantiation
template class Algorithms::AdjointJacobian<StateVectorKokkos<float>>;
template class Algorithms::AdjointJacobian<StateVectorKokkos<double>>;

#include "AdjointDiffKokkos.hpp"
#include "StateVectorKokkos.hpp"

using namespace Pennylane::Lightning_Kokkos;

// explicit instantiation
template class Algorithms::AdjointJacobian<StateVectorKokkos<float>>;
template class Algorithms::AdjointJacobian<StateVectorKokkos<double>>;

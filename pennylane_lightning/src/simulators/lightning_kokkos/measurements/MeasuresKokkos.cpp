#include "MeasuresKokkos.hpp"
#include "StateVectorKokkos.hpp"

using namespace Pennylane::Lightning_Kokkos;

// explicit instantiation
template class Measures::Measurements<StateVectorKokkos<float>>;
template class Measures::Measurements<StateVectorKokkos<double>>;

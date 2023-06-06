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
#pragma once

#include "CPUMemoryModel.hpp" // getAllocator
#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "Error.hpp"
#include "LinearAlgebra.hpp" // scaleAndAdd
#include "Macros.hpp"        // use_openmp
#include "Observables.hpp"
#include "StateVectorLQubitManaged.hpp"
#include "StateVectorLQubitRaw.hpp"
#include "Util.hpp"

#include <complex>
#include <memory>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <iostream>

// using namespace Pennylane;
/// @cond DEV
namespace {
using namespace Pennylane::Util;
using namespace Pennylane::Observables;

using Pennylane::LightningQubit::StateVectorLQubitManaged;
using Pennylane::LightningQubit::StateVectorLQubitRaw;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit::Observables {

/**
 * @brief Final class for named observables (PauliX, PauliY, PauliZ, etc.)
 *
 * @tparam StateVectorT State vector class.
 */
template <class StateVectorT>
class NamedObs final : public NamedObsBase<StateVectorT> {
  public:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using BaseType = NamedObsBase<StateVectorT>;
    /**
     * @brief Construct a NamedObs object, representing a given observable.
     *
     * @param obs_name Name of the observable.
     * @param wires Argument to construct wires.
     * @param params Argument to construct parameters
     */
    NamedObs(std::string obs_name, std::vector<size_t> wires,
             std::vector<PrecisionT> params = {})
        : BaseType{obs_name, wires, params} {
        using Pennylane::LightningQubit::Gates::Constant::gate_names;
        using Pennylane::LightningQubit::Gates::Constant::gate_num_params;
        using Pennylane::LightningQubit::Gates::Constant::gate_wires;
        using Pennylane::LightningQubit::Util::lookup;

        const auto gate_op = lookup(Util::reverse_pairs(gate_names),
                                    std::string_view{this->obs_name_});
        PL_ASSERT(lookup(gate_wires, gate_op) == this->wires_.size());
        PL_ASSERT(lookup(gate_num_params, gate_op) == this->params_.size());
    }
};

/**
 * @brief Final class for Hermitian observables
 *
 * @tparam StateVectorT State vector class.
 */
template <class StateVectorT>
class HermitianObs final : public HermitianObsBase<StateVectorT> {
  public:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using MatrixT = std::vector<std::complex<PrecisionT>>;
    using BaseType = HermitianObsBase<StateVectorT>;

    /**
     * @brief Create an Hermitian observable
     *
     * @param matrix Matrix in row major format.
     * @param wires Wires the observable applies to.
     */
    HermitianObs(MatrixT matrix, std::vector<size_t> wires)
        : BaseType{matrix, wires} {}
};

/**
 * @brief Final class for TensorProdObs observables
 *
 * @tparam StateVectorT State vector class.
 */
template <class StateVectorT>
class TensorProdObs final : public TensorProdObsBase<StateVectorT> {
  public:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using BaseType = TensorProdObsBase<StateVectorT>;
    /**
     * @brief Create an Hermitian observable
     *
     * @param matrix Matrix in row major format.
     * @param wires Wires the observable applies to.
     */
    template <typename... Ts>
    explicit TensorProdObs(Ts &&...arg) : BaseType{arg...} {}

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param obs List of observables
     */
    static auto
    create(std::initializer_list<std::shared_ptr<Observable<StateVectorT>>> obs)
        -> std::shared_ptr<TensorProdObs<StateVectorT>> {
        return std::shared_ptr<TensorProdObs<StateVectorT>>{
            new TensorProdObs(std::move(obs))};
    }

    static auto
    create(std::vector<std::shared_ptr<Observable<StateVectorT>>> obs)
        -> std::shared_ptr<TensorProdObs<StateVectorT>> {
        return std::shared_ptr<TensorProdObs<StateVectorT>>{
            new TensorProdObs(std::move(obs))};
    }
};

/// @cond DEV
namespace detail {
using Pennylane::LightningQubit::Util::scaleAndAdd;

// Default implementation
template <class StateVectorT, bool use_openmp> struct HamiltonianApplyInPlace {
    using PrecisionT = typename StateVectorT::PrecisionT;
    static void run([[maybe_unused]] const std::vector<PrecisionT> &coeffs,
                    [[maybe_unused]] const std::vector<
                        std::shared_ptr<Observable<StateVectorT>>> &terms,
                    [[maybe_unused]] StateVectorT &sv) {
        PL_ABORT("HamiltonianApplyInPlace::run() not implemented for this "
                 "combination of State Vector, Precision and openMP usage.");
    }
};

template <class PrecisionT>
struct HamiltonianApplyInPlace<StateVectorLQubitManaged<PrecisionT>, false> {
    static void
    run(const std::vector<PrecisionT> &coeffs,
        const std::vector<
            std::shared_ptr<Observable<StateVectorLQubitManaged<PrecisionT>>>>
            &terms,
        StateVectorLQubitManaged<PrecisionT> &sv) {
        auto allocator = sv.allocator();
        std::vector<std::complex<PrecisionT>, decltype(allocator)> res(
            sv.getLength(), std::complex<PrecisionT>{0.0, 0.0}, allocator);
        for (size_t term_idx = 0; term_idx < coeffs.size(); term_idx++) {
            StateVectorLQubitManaged<PrecisionT> tmp(sv);
            terms[term_idx]->applyInPlace(tmp);
            Util::scaleAndAdd(tmp.getLength(),
                              std::complex<PrecisionT>{coeffs[term_idx], 0.0},
                              tmp.getData(), res.data());
        }
        sv.updateData(res);
    }
};

#if defined(_OPENMP)
template <class PrecisionT>
struct HamiltonianApplyInPlace<StateVectorLQubitManaged<PrecisionT>, true> {
    static void
    run(const std::vector<PrecisionT> &coeffs,
        const std::vector<
            std::shared_ptr<Observable<StateVectorLQubitManaged<PrecisionT>>>>
            &terms,
        StateVectorLQubitManaged<PrecisionT> &sv) {
        const size_t length = sv.getLength();
        auto allocator = sv.allocator();

        std::vector<std::complex<PrecisionT>, decltype(allocator)> sum(
            length, std::complex<PrecisionT>{}, allocator);

#pragma omp parallel default(none) firstprivate(length, allocator)             \
    shared(coeffs, terms, sv, sum)
        {
            StateVectorLQubitManaged<PrecisionT> tmp(sv.getNumQubits());

            std::vector<std::complex<PrecisionT>, decltype(allocator)> local_sv(
                length, std::complex<PrecisionT>{}, allocator);

#pragma omp for
            for (size_t term_idx = 0; term_idx < terms.size(); term_idx++) {
                tmp.updateData(sv.getDataVector());
                terms[term_idx]->applyInPlace(tmp);
                scaleAndAdd(length,
                            std::complex<PrecisionT>{coeffs[term_idx], 0.0},
                            tmp.getData(), local_sv.data());
            }

#pragma omp critical
            {
                scaleAndAdd(length, std::complex<PrecisionT>{1.0, 0.0},
                            local_sv.data(), sum.data());
            }
        }

        sv.updateData(sum);
    }
};

#endif

} // namespace detail
/// @endcond

/**
 * @brief Final class for a general Hamiltonian representation as a sum of
 * observables.
 *
 * @tparam StateVectorT State vector class.
 */
template <class StateVectorT>
class Hamiltonian final : public HamiltonianBase<StateVectorT> {
  public:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using BaseType = HamiltonianBase<StateVectorT>;

    /**
     * @brief Create a Hamiltonian from coefficients and observables
     *
     * @param coeffs Arguments to construct coefficients
     * @param obs Arguments to construct observables
     */
    template <typename T1, typename T2>
    explicit Hamiltonian(T1 &&coeffs, T2 &&obs) : BaseType{coeffs, obs} {}

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param coeffs Arguments to construct coefficients
     * @param obs Arguments to construct observables
     */
    static auto
    create(std::initializer_list<PrecisionT> coeffs,
           std::initializer_list<std::shared_ptr<Observable<StateVectorT>>> obs)
        -> std::shared_ptr<Hamiltonian<StateVectorT>> {
        return std::shared_ptr<Hamiltonian<StateVectorT>>(
            new Hamiltonian<StateVectorT>{std::move(coeffs), std::move(obs)});
    }

    void applyInPlace(StateVectorT &sv) const override {
        detail::HamiltonianApplyInPlace<
            StateVectorT, Pennylane::Util::use_openmp>::run(this->coeffs_,
                                                            this->obs_, sv);
    }
};

} // namespace Pennylane::LightningQubit::Observables
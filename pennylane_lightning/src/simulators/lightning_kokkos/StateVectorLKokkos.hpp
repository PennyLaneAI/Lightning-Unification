// Copyright 2023 Xanadu Quantum Technologies Inc.

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
 * @file StateVectorLKokkos.hpp
 */

#pragma once
#include <complex>
#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "BitUtil.hpp"
#include "Error.hpp"
#include "StateVectorBase.hpp"
#include "Util.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Util;
} // namespace
/// @endcond

namespace Pennylane {

/**
 * @brief Kokkos functor for initializing the state vector to the \f$\ket{0}\f$
 * state
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 */
template <typename PrecisionT> struct InitView {
    Kokkos::View<Kokkos::complex<PrecisionT> *> a;
    InitView(Kokkos::View<Kokkos::complex<PrecisionT> *> a_) : a(a_) {}
    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t i) const {
        a(i) = Kokkos::complex<PrecisionT>((i == 0) * 1.0, 0.0);
    }
};

/**
 * @brief Kokkos functor for setting the basis state
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 */
template <typename PrecisionT> struct setBasisStateFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> a;
    const std::size_t index;
    setBasisStateFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> a_,
                         const std::size_t index_)
        : a(a_), index(index_) {}
    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t i) const {
        a(i) = Kokkos::complex<PrecisionT>((i == index) * 1.0, 0.0);
    }
};

/**
 * @brief Kokkos functor for setting the state vector
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 */
template <typename PrecisionT> struct setStateVectorFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> a;
    Kokkos::View<size_t *> indices;
    Kokkos::View<Kokkos::complex<PrecisionT> *> values;
    setStateVectorFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> a_,
        const Kokkos::View<size_t *> indices_,
        const Kokkos::View<Kokkos::complex<PrecisionT> *> values_)
        : a(a_), indices(indices_), values(values_) {}
    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t i) const { a(indices[i]) = values[i]; }
};

/**
 * @brief Kokkos functor for initializing zeros to the state vector.
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 */
template <typename PrecisionT> struct initZerosFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> a;
    initZerosFunctor(Kokkos::View<Kokkos::complex<PrecisionT> *> a_) : a(a_) {}
    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t i) const {
        a(i) = Kokkos::complex<PrecisionT>(0.0, 0.0);
    }
};

/**
 * @brief  Kokkos state vector class
 *
 * @tparam PrecisionT Floating-point precision type.
 */
template <class PrecisionT = double>
class StateVectorLKokkos
    : public StateVectorBase<PrecisionT, StateVectorLKokkos<PrecisionT>> {
  public:
    using BaseType =
        StateVectorBase<PrecisionT, StateVectorLKokkos<PrecisionT>>;
    using KokkosVector = Kokkos::View<Kokkos::complex<PrecisionT> *>;

  private:
    size_t length_;
    std::mutex init_mutex_;
    std::unique_ptr<KokkosVector> data_;
    inline static bool is_exit_reg_ = false;

  public:
    using KokkosExecSpace = Kokkos::DefaultExecutionSpace;
    using KokkosSizeTVector = Kokkos::View<size_t *>;
    using KokkosRangePolicy = Kokkos::RangePolicy<KokkosExecSpace>;
    using UnmanagedComplexHostView =
        Kokkos::View<Kokkos::complex<PrecisionT> *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UnmanagedSizeTHostView =
        Kokkos::View<size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UnmanagedConstComplexHostView =
        Kokkos::View<const Kokkos::complex<PrecisionT> *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UnmanagedConstSizeTHostView =
        Kokkos::View<const size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    StateVectorLKokkos() = delete;
    StateVectorLKokkos(size_t num_qubits,
                       const Kokkos::InitializationSettings &kokkos_args = {})
        : BaseType{num_qubits}, length_(Util::exp2(num_qubits)) {
        {
            const std::lock_guard<std::mutex> lock(init_mutex_);
            if (!Kokkos::is_initialized()) {
                Kokkos::initialize(kokkos_args);
            }
        }

        if (num_qubits > 0) {
            data_ =
                std::make_unique<KokkosVector>("data_", Util::exp2(num_qubits));
            Kokkos::parallel_for(length_, InitView(*data_));
        }
    };

    /**
     * @brief Create a new state vector from data on the host.
     *
     * @param num_qubits Number of qubits
     */
    StateVectorLKokkos(std::complex<PrecisionT> *hostdata_, size_t length,
                       const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorLKokkos(Util::log2PerfectPower(length), kokkos_args) {
        // check if length is a power of 2.
        if (!Util::isPerfectPowerOf2(length)) {
            PL_ABORT(
                "The length of the state vector must be a power of 2. But " +
                std::to_string(length) +
                " was given."); // TODO: change to std::format in C++20
        }
        HostToDevice(reinterpret_cast<Kokkos::complex<PrecisionT> *>(hostdata_),
                     length);
    }

    /**
     * @brief Copy constructor
     *
     * @param other Another Kokkos state vector
     */
    StateVectorLKokkos(const StateVectorLKokkos &other,
                       const Kokkos::InitializationSettings &kokkos_args = {})
        : StateVectorLKokkos(other.getNumQubits(), kokkos_args) {
        this->DeviceToDevice(other.getData());
    }

    /**
     * @brief Reset the data back to the \f$\ket{0}\f$ state.
     *
     * @param num_qubits Number of qubits
     */
    void resetStateVector() {
        if (length_ > 0) {
            Kokkos::parallel_for(length_, InitView(*data_));
        }
    }

    /**
     * @brief Destructor for StateVectorLKokkos class
     *
     * @param other Another state vector
     */
    ~StateVectorLKokkos() {
        data_.reset();
        {
            const std::lock_guard<std::mutex> lock(init_mutex_);
            if (!is_exit_reg_) {
                is_exit_reg_ = true;
                std::atexit([]() {
                    if (!Kokkos::is_finalized()) {
                        Kokkos::finalize();
                    }
                });
            }
        }
    }

    /**
     * @brief Get the size of the state vector
     *
     * @return The size of the state vector
     */
    size_t getLength() const { return length_; }

    /**
     * @brief Update state vector with data from other state vector.
     *
     * @param other
     */
    void updateData(const StateVectorLKokkos<PrecisionT> &other) {
        Kokkos::deep_copy(*data_, other.getData());
    }

    /**
     * @brief Get the Kokkos data of the state vector.
     *
     * @return The pointer to the data of state vector
     */
    [[nodiscard]] auto getData() const -> KokkosVector & { return *data_; }

    /**
     * @brief Get the Kokkos data of the state vector
     *
     * @return The pointer to the data of state vector
     */
    [[nodiscard]] auto getData() -> KokkosVector & { return *data_; }

    /**
     * @brief Copy data from the host space to the device space.
     *
     */
    inline void HostToDevice(Kokkos::complex<PrecisionT> *sv, size_t length) {
        Kokkos::deep_copy(*data_, UnmanagedComplexHostView(sv, length));
    }

    /**
     * @brief Copy data from the device space to the host space.
     *
     */
    inline void DeviceToHost(Kokkos::complex<PrecisionT> *sv, size_t length) {
        Kokkos::deep_copy(UnmanagedComplexHostView(sv, length), *data_);
    }

    /**
     * @brief Copy data from the device space to the device space.
     *
     */
    inline void DeviceToDevice(KokkosVector vector_to_copy) {
        Kokkos::deep_copy(*data_, vector_to_copy);
    }

    /**
     * @brief Compare two PennyLane Lightning Kokkos state vectors.
     *
     * @param rhs The second state vector.
     * @return true
     * @return false
     */
    bool operator==(const StateVectorLKokkos<PrecisionT> &rhs) const {
        if (BaseType::getNumQubits() != rhs.getNumQubits()) {
            return false;
        }
        const KokkosVector data1 = getData();
        const KokkosVector data2 = rhs.getData();
        for (size_t k = 0; k < getLength(); k++) {
            if (data1[k] != data2[k]) {
                return false;
            }
        }
        return true;
    }
};

}; // namespace Pennylane
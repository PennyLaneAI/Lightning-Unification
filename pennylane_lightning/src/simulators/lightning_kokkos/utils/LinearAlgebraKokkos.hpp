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
 * @file LinearAlgebraKokkos.hpp
 * Contains uncategorised utility functions.
 */

#pragma once

#include <Kokkos_Core.hpp>

namespace Pennylane::LightningKokkos::Util {

/**
 * @brief @rst
 * Kokkos functor for :math:`y+=\alpha*x` operation.
 * @endrst
 */
template <class PrecisionT> struct axpy_KokkosFunctor {
    Kokkos::complex<PrecisionT> alpha;
    Kokkos::View<Kokkos::complex<PrecisionT> *> x;
    Kokkos::View<Kokkos::complex<PrecisionT> *> y;
    axpy_KokkosFunctor(Kokkos::complex<PrecisionT> alpha_,
                       Kokkos::View<Kokkos::complex<PrecisionT> *> x_,
                       Kokkos::View<Kokkos::complex<PrecisionT> *> y_) {
        alpha = alpha_;
        x = x_;
        y = y_;
    }
    KOKKOS_INLINE_FUNCTION void operator()(const std::size_t k) const {
        y[k] += alpha * x[k];
    }
};

/**
 * @brief @rst
 * Kokkos implementation of the :math:`y+=\alpha*x` operation.
 * @endrst
 * @param alpha Scalar to scale x
 * @param x Vector to add
 * @param y Vector to be added
 * @param length number of elements in x
 * */
template <class PrecisionT>
inline auto axpy_Kokkos(Kokkos::complex<PrecisionT> alpha,
                        Kokkos::View<Kokkos::complex<PrecisionT> *> x,
                        Kokkos::View<Kokkos::complex<PrecisionT> *> y,
                        std::size_t length) {
    Kokkos::parallel_for(length, axpy_KokkosFunctor<PrecisionT>(alpha, x, y));
}

/**
 * @brief @rst
 * Sparse matrix vector multiply functor :math: `y=A*x`.
 * @endrst
 */
template <class PrecisionT> struct SparseMV_KokkosFunctor {

    using KokkosVector = Kokkos::View<Kokkos::complex<PrecisionT> *>;
    using KokkosSizeTVector = Kokkos::View<std::size_t *>;

    KokkosVector x;
    KokkosVector y;
    KokkosVector data;
    KokkosSizeTVector indices;
    KokkosSizeTVector indptr;

    SparseMV_KokkosFunctor(KokkosVector x_, KokkosVector y_,
                           const KokkosVector data_,
                           const KokkosSizeTVector indices_,
                           const KokkosSizeTVector indptr_) {
        x = x_;
        y = y_;
        data = data_;
        indices = indices_;
        indptr = indptr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t row) const {
        Kokkos::complex<PrecisionT> tmp = {0.0, 0.0};
        for (size_t j = indptr[row]; j < indptr[row + 1]; j++) {
            tmp += data[j] * x[indices[j]];
        }
        y[row] = tmp;
    }
};

/**
 * @brief @rst
 * Sparse matrix vector multiply :math: `y=A*x`.
 * @endrst
 * @param x Input vector
 * @param y Result vector
 * @param data Vector of non-zeros elements of the sparse matrix A.
 * @param indices Vector of column indices of data.
 * @param indptr Vector of offsets.
 */
template <class PrecisionT>
inline void SparseMV_Kokkos(Kokkos::View<Kokkos::complex<PrecisionT> *> x,
                            Kokkos::View<Kokkos::complex<PrecisionT> *> y,
                            const std::vector<std::complex<PrecisionT>> &data,
                            const std::vector<std::size_t> &indices,
                            const std::vector<std::size_t> &indptr) {

    using ConstComplexHostView =
        Kokkos::View<const Kokkos::complex<PrecisionT> *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using ConstSizeTHostView =
        Kokkos::View<const size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using KokkosSizeTVector = Kokkos::View<size_t *>;
    using KokkosVector = Kokkos::View<Kokkos::complex<PrecisionT> *>;

    KokkosVector kok_data("kokkos_sparese_matrix_vals", data.size());
    KokkosSizeTVector kok_indices("kokkos_indices", indices.size());
    KokkosSizeTVector kok_indptr("kokkos_offsets", indptr.size());

    auto data_ptr =
        reinterpret_cast<const Kokkos::complex<PrecisionT> *>(data.data());

    const std::vector<Kokkos::complex<PrecisionT>> kok_complex_data =
        std::vector<Kokkos::complex<PrecisionT>>{data_ptr,
                                                data_ptr + data.size()};

    Kokkos::deep_copy(
        kok_data, ConstComplexHostView(kok_complex_data.data(), data.size()));

    Kokkos::deep_copy(kok_indices,
                      ConstSizeTHostView(indices.data(), indices.size()));
    Kokkos::deep_copy(kok_indptr,
                      ConstSizeTHostView(indptr.data(), indptr.size()));

    Kokkos::parallel_for(indptr.size() - 1,
                         SparseMV_KokkosFunctor<PrecisionT>(
                             x, y, kok_data, kok_indices, kok_indptr));
}

/**
 * @brief @rst
 * Kokkos functor of the :math:`real(conj(x)*y)` operation.
 * @endrst
 */
template <class PrecisionT> struct getRealOfComplexInnerProductFunctor {

    Kokkos::View<Kokkos::complex<PrecisionT> *> x;
    Kokkos::View<Kokkos::complex<PrecisionT> *> y;

    getRealOfComplexInnerProductFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> x_,
        Kokkos::View<Kokkos::complex<PrecisionT> *> y_) {
        x = x_;
        y = y_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k, PrecisionT &inner) const {
        inner += real(x[k]) * real(y[k]) + imag(x[k]) * imag(y[k]);
    }
};

/**
 * @brief @rst
 * Kokkos implementation of the :math:`real(conj(x)*y)` operation.
 * @endrst
 * @param x Input vector
 * @param y Input vector
 * @return :math:`real(conj(x)*y)`
 */
template <class PrecisionT>
inline auto
getRealOfComplexInnerProduct(Kokkos::View<Kokkos::complex<PrecisionT> *> x,
                             Kokkos::View<Kokkos::complex<PrecisionT> *> y)
    -> PrecisionT {

    assert(x.size() == y.size());
    PrecisionT inner = 0;
    Kokkos::parallel_reduce(
        x.size(), getRealOfComplexInnerProductFunctor<PrecisionT>(x, y), inner);
    return inner;
}

/**
 * @brief @rstimagine
 * Kokkos functor of the :math:`imag(conj(x)*y)` operation.
 * @endrst
 */
template <class PrecisionT> struct getImagOfComplexInnerProductFunctor {

    Kokkos::View<Kokkos::complex<PrecisionT> *> x;
    Kokkos::View<Kokkos::complex<PrecisionT> *> y;

    getImagOfComplexInnerProductFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> x_,
        Kokkos::View<Kokkos::complex<PrecisionT> *> y_) {
        x = x_;
        y = y_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k, PrecisionT &inner) const {
        inner += real(x[k]) * imag(y[k]) - imag(x[k]) * real(y[k]);
    }
};

/**
 * @brief @rst
 * Kokkos implementation of the :math:`imag(conj(x)*y)` operation.
 * @endrst
 * @param x Input vector
 * @param y Input vector
 * @return :math:`imag(conj(x)*y)`
 */
template <class PrecisionT>
inline auto
getImagOfComplexInnerProduct(Kokkos::View<Kokkos::complex<PrecisionT> *> x,
                             Kokkos::View<Kokkos::complex<PrecisionT> *> y)
    -> PrecisionT {

    assert(x.size() == y.size());
    PrecisionT inner = 0;
    Kokkos::parallel_reduce(
        x.size(), getImagOfComplexInnerProductFunctor<PrecisionT>(x, y), inner);
    return inner;
}

} // namespace Pennylane::LightningKokkos::Util

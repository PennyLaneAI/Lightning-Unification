#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "BitUtil.hpp"

namespace {
using namespace Pennylane::Util;
using Kokkos::Experimental::swap;
} // namespace

namespace Pennylane::LightningKokkos::Functors {
template <class Precision, bool adj = false> struct generatorPhaseShiftFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire;
    std::size_t rev_wire_shift;
    std::size_t wire_parity;
    std::size_t wire_parity_inv;

    generatorPhaseShiftFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<Precision> &params) {
        arr = arr_;
        rev_wire = num_qubits - wires[0] - 1;
        rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        wire_parity = fillTrailingOnes(rev_wire);
        wire_parity_inv = fillLeadingOnes(rev_wire + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i0 =
            ((k << 1U) & wire_parity_inv) | (wire_parity & k);
        arr[i0] = Kokkos::complex<Precision>{0.0, 0.0};
    }
};

template <class Precision, bool adj = false> struct generatorIsingXXFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    generatorIsingXXFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<Precision> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        swap(arr[i00], arr[i11]);
        swap(arr[i10], arr[i01]);
    }
};

template <class Precision, bool adj = false> struct generatorIsingXYFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    generatorIsingXYFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<Precision> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        swap(arr[i10], arr[i01]);
        arr[i00] = Kokkos::complex<Precision>{0.0, 0.0};
        arr[i11] = Kokkos::complex<Precision>{0.0, 0.0};
    }
};

template <class Precision, bool adj = false> struct generatorIsingYYFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    generatorIsingYYFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<Precision> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const auto v00 = arr[i00];
        arr[i00] = -arr[i11];
        arr[i11] = -v00;
        swap(arr[i10], arr[i01]);
    }
};

template <class Precision, bool adj = false> struct generatorIsingZZFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    generatorIsingZZFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<Precision> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);
        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i10 = i00 | rev_wire1_shift;

        arr[i10] *= -1;
        arr[i01] *= -1;
    }
};

template <class Precision, bool adj = false>
struct generatorSingleExcitationFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    generatorSingleExcitationFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<Precision> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        using ComplexPrecision = Kokkos::complex<Precision>;
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        arr[i00] = ComplexPrecision{};
        arr[i01] *= ComplexPrecision{0, 1};
        arr[i10] *= ComplexPrecision{0, -1};
        arr[i11] = ComplexPrecision{};
        swap(arr[i10], arr[i01]);
    }
};

template <class Precision, bool adj = false>
struct generatorSingleExcitationMinusFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    generatorSingleExcitationMinusFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<Precision> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        using ComplexPrecision = Kokkos::complex<Precision>;
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i10 = i00 | rev_wire1_shift;

        arr[i01] *= ComplexPrecision{0, 1};
        arr[i10] *= ComplexPrecision{0, -1};
        swap(arr[i10], arr[i01]);
    }
};

template <class Precision, bool adj = false>
struct generatorSingleExcitationPlusFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    generatorSingleExcitationPlusFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<Precision> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        using ComplexPrecision = Kokkos::complex<Precision>;
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        arr[i00] *= -1;
        arr[i01] *= ComplexPrecision{0, 1};
        arr[i10] *= ComplexPrecision{0, -1};
        arr[i11] *= -1;

        swap(arr[i10], arr[i01]);
    }
};

template <class Precision, bool inverse = false>
struct generatorDoubleExcitationFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire2;
    std::size_t rev_wire3;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire2_shift;
    std::size_t rev_wire3_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_min_mid;
    std::size_t rev_wire_max_mid;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;
    std::size_t parity_hmiddle;
    std::size_t parity_lmiddle;

    Kokkos::complex<Precision> shifts_0;
    Kokkos::complex<Precision> shifts_1;
    Kokkos::complex<Precision> shifts_2;
    Kokkos::complex<Precision> shifts_3;

    generatorDoubleExcitationFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<Precision> &params) {
        rev_wire0 = num_qubits - wires[3] - 1;
        rev_wire1 = num_qubits - wires[2] - 1;
        rev_wire2 = num_qubits - wires[1] - 1;
        rev_wire3 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        rev_wire2_shift = static_cast<size_t>(1U) << rev_wire2;
        rev_wire3_shift = static_cast<size_t>(1U) << rev_wire3;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_min_mid = std::max(rev_wire0, rev_wire1);
        rev_wire_max_mid = std::min(rev_wire2, rev_wire3);
        rev_wire_max = std::max(rev_wire2, rev_wire3);

        if (rev_wire_max_mid > rev_wire_min_mid) {
        } else if (rev_wire_max_mid < rev_wire_min) {
            if (rev_wire_max < rev_wire_min) {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = tmp;

                tmp = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else if (rev_wire_max > rev_wire_min_mid) {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            }
        } else {
            if (rev_wire_max > rev_wire_min_mid) {
                std::size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = tmp;
            } else {
                std::size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = tmp;
            }
        }

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_lmiddle = fillLeadingOnes(rev_wire_min + 1) &
                         fillTrailingOnes(rev_wire_min_mid);
        parity_hmiddle = fillLeadingOnes(rev_wire_max_mid + 1) &
                         fillTrailingOnes(rev_wire_max);
        parity_middle = fillLeadingOnes(rev_wire_min_mid + 1) &
                        fillTrailingOnes(rev_wire_max_mid);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        using ComplexPrecision = Kokkos::complex<Precision>;
        const std::size_t i0000 =
            ((k << 4U) & parity_high) | ((k << 3U) & parity_hmiddle) |
            ((k << 2U) & parity_middle) | ((k << 1U) & parity_lmiddle) |
            (k & parity_low);
        const std::size_t i0001 = i0000 | rev_wire0_shift;
        const std::size_t i0010 = i0000 | rev_wire1_shift;
        const std::size_t i0011 = i0000 | rev_wire1_shift | rev_wire0_shift;
        const std::size_t i0100 = i0000 | rev_wire2_shift;
        const std::size_t i0101 = i0000 | rev_wire2_shift | rev_wire0_shift;
        const std::size_t i0110 = i0000 | rev_wire2_shift | rev_wire1_shift;
        const std::size_t i0111 =
            i0000 | rev_wire2_shift | rev_wire1_shift | rev_wire0_shift;
        const std::size_t i1000 = i0000 | rev_wire3_shift;
        const std::size_t i1001 = i0000 | rev_wire3_shift | rev_wire0_shift;
        const std::size_t i1010 = i0000 | rev_wire3_shift | rev_wire1_shift;
        const std::size_t i1011 =
            i0000 | rev_wire3_shift | rev_wire1_shift | rev_wire0_shift;
        const std::size_t i1100 = i0000 | rev_wire3_shift | rev_wire2_shift;
        const std::size_t i1101 =
            i0000 | rev_wire3_shift | rev_wire2_shift | rev_wire0_shift;
        const std::size_t i1110 =
            i0000 | rev_wire3_shift | rev_wire2_shift | rev_wire1_shift;
        const std::size_t i1111 = i0000 | rev_wire3_shift | rev_wire2_shift |
                                  rev_wire1_shift | rev_wire0_shift;

        const Kokkos::complex<Precision> v3 = arr[i0011];
        const Kokkos::complex<Precision> v12 = arr[i1100];
        arr[i0000] = ComplexPrecision{};
        arr[i0001] = ComplexPrecision{};
        arr[i0010] = ComplexPrecision{};
        arr[i0011] = v12 * ComplexPrecision{0, -1};
        arr[i0100] = ComplexPrecision{};
        arr[i0101] = ComplexPrecision{};
        arr[i0110] = ComplexPrecision{};
        arr[i0111] = ComplexPrecision{};
        arr[i1000] = ComplexPrecision{};
        arr[i1001] = ComplexPrecision{};
        arr[i1010] = ComplexPrecision{};
        arr[i1011] = ComplexPrecision{};
        arr[i1100] = v3 * ComplexPrecision{0, 1};
        arr[i1101] = ComplexPrecision{};
        arr[i1110] = ComplexPrecision{};
        arr[i1111] = ComplexPrecision{};
    }
};

template <class Precision, bool inverse = false>
struct generatorDoubleExcitationMinusFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire2;
    std::size_t rev_wire3;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire2_shift;
    std::size_t rev_wire3_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_min_mid;
    std::size_t rev_wire_max_mid;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;
    std::size_t parity_hmiddle;
    std::size_t parity_lmiddle;

    Kokkos::complex<Precision> shifts_0;
    Kokkos::complex<Precision> shifts_1;
    Kokkos::complex<Precision> shifts_2;
    Kokkos::complex<Precision> shifts_3;

    generatorDoubleExcitationMinusFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<Precision> &params) {
        rev_wire0 = num_qubits - wires[3] - 1;
        rev_wire1 = num_qubits - wires[2] - 1;
        rev_wire2 = num_qubits - wires[1] - 1;
        rev_wire3 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        rev_wire2_shift = static_cast<size_t>(1U) << rev_wire2;
        rev_wire3_shift = static_cast<size_t>(1U) << rev_wire3;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_min_mid = std::max(rev_wire0, rev_wire1);
        rev_wire_max_mid = std::min(rev_wire2, rev_wire3);
        rev_wire_max = std::max(rev_wire2, rev_wire3);

        if (rev_wire_max_mid > rev_wire_min_mid) {
        } else if (rev_wire_max_mid < rev_wire_min) {
            if (rev_wire_max < rev_wire_min) {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = tmp;

                tmp = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else if (rev_wire_max > rev_wire_min_mid) {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            }
        } else {
            if (rev_wire_max > rev_wire_min_mid) {
                std::size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = tmp;
            } else {
                std::size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = tmp;
            }
        }

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_lmiddle = fillLeadingOnes(rev_wire_min + 1) &
                         fillTrailingOnes(rev_wire_min_mid);
        parity_hmiddle = fillLeadingOnes(rev_wire_max_mid + 1) &
                         fillTrailingOnes(rev_wire_max);
        parity_middle = fillLeadingOnes(rev_wire_min_mid + 1) &
                        fillTrailingOnes(rev_wire_max_mid);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        using ComplexPrecision = Kokkos::complex<Precision>;
        const std::size_t i0000 =
            ((k << 4U) & parity_high) | ((k << 3U) & parity_hmiddle) |
            ((k << 2U) & parity_middle) | ((k << 1U) & parity_lmiddle) |
            (k & parity_low);
        const std::size_t i0011 = i0000 | rev_wire1_shift | rev_wire0_shift;
        const std::size_t i1100 = i0000 | rev_wire3_shift | rev_wire2_shift;

        arr[i0011] *= ComplexPrecision{0, 1};
        arr[i1100] *= ComplexPrecision{0, -1};
        swap(arr[i1100], arr[i0011]);
    }
};

template <class Precision, bool inverse = false>
struct generatorDoubleExcitationPlusFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire2;
    std::size_t rev_wire3;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire2_shift;
    std::size_t rev_wire3_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_min_mid;
    std::size_t rev_wire_max_mid;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;
    std::size_t parity_hmiddle;
    std::size_t parity_lmiddle;

    Kokkos::complex<Precision> shifts_0;
    Kokkos::complex<Precision> shifts_1;
    Kokkos::complex<Precision> shifts_2;
    Kokkos::complex<Precision> shifts_3;

    generatorDoubleExcitationPlusFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<Precision> &params) {
        rev_wire0 = num_qubits - wires[3] - 1;
        rev_wire1 = num_qubits - wires[2] - 1;
        rev_wire2 = num_qubits - wires[1] - 1;
        rev_wire3 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
        rev_wire2_shift = static_cast<size_t>(1U) << rev_wire2;
        rev_wire3_shift = static_cast<size_t>(1U) << rev_wire3;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_min_mid = std::max(rev_wire0, rev_wire1);
        rev_wire_max_mid = std::min(rev_wire2, rev_wire3);
        rev_wire_max = std::max(rev_wire2, rev_wire3);

        if (rev_wire_max_mid > rev_wire_min_mid) {
        } else if (rev_wire_max_mid < rev_wire_min) {
            if (rev_wire_max < rev_wire_min) {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = tmp;

                tmp = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else if (rev_wire_max > rev_wire_min_mid) {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            } else {
                std::size_t tmp = rev_wire_min;
                rev_wire_min = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = rev_wire_min_mid;
                rev_wire_min_mid = tmp;
            }
        } else {
            if (rev_wire_max > rev_wire_min_mid) {
                std::size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = tmp;
            } else {
                std::size_t tmp = rev_wire_min_mid;
                rev_wire_min_mid = rev_wire_max_mid;
                rev_wire_max_mid = rev_wire_max;
                rev_wire_max = tmp;
            }
        }

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_lmiddle = fillLeadingOnes(rev_wire_min + 1) &
                         fillTrailingOnes(rev_wire_min_mid);
        parity_hmiddle = fillLeadingOnes(rev_wire_max_mid + 1) &
                         fillTrailingOnes(rev_wire_max);
        parity_middle = fillLeadingOnes(rev_wire_min_mid + 1) &
                        fillTrailingOnes(rev_wire_max_mid);

        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        using ComplexPrecision = Kokkos::complex<Precision>;
        const std::size_t i0000 =
            ((k << 4U) & parity_high) | ((k << 3U) & parity_hmiddle) |
            ((k << 2U) & parity_middle) | ((k << 1U) & parity_lmiddle) |
            (k & parity_low);
        const std::size_t i0011 = i0000 | rev_wire1_shift | rev_wire0_shift;
        const std::size_t i1100 = i0000 | rev_wire3_shift | rev_wire2_shift;

        arr[i0011] *= ComplexPrecision{0, -1};
        arr[i1100] *= ComplexPrecision{0, 1};
        swap(arr[i1100], arr[i0011]);
    }
};

template <class Precision, bool adj = false>
struct generatorControlledPhaseShiftFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    generatorControlledPhaseShiftFunctor(
        Kokkos::View<Kokkos::complex<Precision> *> &arr_,
        std::size_t num_qubits, const std::vector<size_t> &wires,
        [[maybe_unused]] const std::vector<Precision> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);
        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i10 = i00 | rev_wire1_shift;

        arr[i00] = Kokkos::complex<Precision>{0.0, 0.0};
        arr[i01] = Kokkos::complex<Precision>{0.0, 0.0};
        arr[i10] = Kokkos::complex<Precision>{0.0, 0.0};
    }
};

template <class Precision, bool adj = false> struct generatorCRXFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    generatorCRXFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                        std::size_t num_qubits,
                        const std::vector<size_t> &wires,
                        [[maybe_unused]] const std::vector<Precision> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);
        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        arr[i00] = Kokkos::complex<Precision>{0.0, 0.0};
        arr[i01] = Kokkos::complex<Precision>{0.0, 0.0};
        swap(arr[i10], arr[i11]);
    }
};

template <class Precision, bool adj = false> struct generatorCRYFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    generatorCRYFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                        std::size_t num_qubits,
                        const std::vector<size_t> &wires,
                        [[maybe_unused]] const std::vector<Precision> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);
        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        using ComplexPrecision = Kokkos::complex<Precision>;
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i10 = i00 | rev_wire1_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        arr[i00] = ComplexPrecision{};
        arr[i01] = ComplexPrecision{};

        const auto v0 = arr[i10];

        arr[i10] = ComplexPrecision{imag(arr[i11]), -real(arr[i11])};
        arr[i11] = ComplexPrecision{-imag(v0), real(v0)};
    }
};

template <class Precision, bool adj = false> struct generatorCRZFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t rev_wire0;
    std::size_t rev_wire1;
    std::size_t rev_wire0_shift;
    std::size_t rev_wire1_shift;
    std::size_t rev_wire_min;
    std::size_t rev_wire_max;
    std::size_t parity_low;
    std::size_t parity_high;
    std::size_t parity_middle;

    generatorCRZFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                        std::size_t num_qubits,
                        const std::vector<size_t> &wires,
                        [[maybe_unused]] const std::vector<Precision> &params) {
        rev_wire0 = num_qubits - wires[1] - 1;
        rev_wire1 = num_qubits - wires[0] - 1; // Control qubit

        rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        rev_wire_min = std::min(rev_wire0, rev_wire1);
        rev_wire_max = std::max(rev_wire0, rev_wire1);

        parity_low = fillTrailingOnes(rev_wire_min);
        parity_high = fillLeadingOnes(rev_wire_max + 1);
        parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);
        arr = arr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        using ComplexPrecision = Kokkos::complex<Precision>;
        const std::size_t i00 = ((k << 2U) & parity_high) |
                                ((k << 1U) & parity_middle) | (k & parity_low);
        const std::size_t i01 = i00 | rev_wire0_shift;
        const std::size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        arr[i00] = ComplexPrecision{};
        arr[i01] = ComplexPrecision{};
        arr[i11] *= -1;
    }
};

template <class Precision, bool adj = false> struct generatorMultiRZFunctor {

    Kokkos::View<Kokkos::complex<Precision> *> arr;

    std::size_t wires_parity;

    generatorMultiRZFunctor(Kokkos::View<Kokkos::complex<Precision> *> &arr_,
                            std::size_t num_qubits,
                            const std::vector<size_t> &wires) {
        arr = arr_;
        wires_parity = static_cast<size_t>(0U);
        for (size_t wire : wires) {
            wires_parity |=
                (static_cast<size_t>(1U) << (num_qubits - wire - 1));
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        arr[k] *= static_cast<Precision>(
            1 - 2 * int(Kokkos::Impl::bit_count(k & wires_parity) % 2));
    }
};
} // namespace Pennylane::LightningKokkos::Functors

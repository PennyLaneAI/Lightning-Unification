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
 * @file
 * Defines helper methods for PennyLane Lightning.
 */
#pragma once

#include "TypeTraits.hpp"

#include <catch2/catch.hpp>

namespace Pennylane::Util {
template <class T, class Alloc = std::allocator<T>> struct PLApprox {
    const std::vector<T, Alloc> &comp_;

    explicit PLApprox(const std::vector<T, Alloc> &comp) : comp_{comp} {}

    Util::remove_complex_t<T> margin_{};
    Util::remove_complex_t<T> epsilon_ =
        std::numeric_limits<float>::epsilon() * 100;

    template <class AllocA>
    [[nodiscard]] bool compare(const std::vector<T, AllocA> &lhs) const {
        if (lhs.size() != comp_.size()) {
            return false;
        }

        for (size_t i = 0; i < lhs.size(); i++) {
            if constexpr (Util::is_complex_v<T>) {
                if (lhs[i].real() != Approx(comp_[i].real())
                                         .epsilon(epsilon_)
                                         .margin(margin_) ||
                    lhs[i].imag() != Approx(comp_[i].imag())
                                         .epsilon(epsilon_)
                                         .margin(margin_)) {
                    return false;
                }
            } else {
                if (lhs[i] !=
                    Approx(comp_[i]).epsilon(epsilon_).margin(margin_)) {
                    return false;
                }
            }
        }
        return true;
    }

    [[nodiscard]] std::string describe() const {
        std::ostringstream ss;
        ss << "is Approx to {";
        for (const auto &elt : comp_) {
            ss << elt << ", ";
        }
        ss << "}" << std::endl;
        return ss.str();
    }

    PLApprox &epsilon(Util::remove_complex_t<T> eps) {
        epsilon_ = eps;
        return *this;
    }
    PLApprox &margin(Util::remove_complex_t<T> m) {
        margin_ = m;
        return *this;
    }
};

/**
 * @brief Simple helper for PLApprox for the cases when the class template
 * deduction does not work well.
 */
template <typename T, class Alloc>
PLApprox<T, Alloc> approx(const std::vector<T, Alloc> &vec) {
    return PLApprox<T, Alloc>(vec);
}

template <typename T, class Alloc>
std::ostream &operator<<(std::ostream &os, const PLApprox<T, Alloc> &approx) {
    os << approx.describe();
    return os;
}
template <class T, class AllocA, class AllocB>
bool operator==(const std::vector<T, AllocA> &lhs,
                const PLApprox<T, AllocB> &rhs) {
    return rhs.compare(lhs);
}
template <class T, class AllocA, class AllocB>
bool operator!=(const std::vector<T, AllocA> &lhs,
                const PLApprox<T, AllocB> &rhs) {
    return !rhs.compare(lhs);
}

template <class PrecisionT> struct PLApproxComplex {
    const std::complex<PrecisionT> comp_;

    explicit PLApproxComplex(const std::complex<PrecisionT> &comp)
        : comp_{comp} {}

    PrecisionT margin_{};
    PrecisionT epsilon_ = std::numeric_limits<float>::epsilon() * 100;

    [[nodiscard]] bool compare(const std::complex<PrecisionT> &lhs) const {
        return (lhs.real() ==
                Approx(comp_.real()).epsilon(epsilon_).margin(margin_)) &&
               (lhs.imag() ==
                Approx(comp_.imag()).epsilon(epsilon_).margin(margin_));
    }
    [[nodiscard]] std::string describe() const {
        std::ostringstream ss;
        ss << "is Approx to " << comp_;
        return ss.str();
    }
    PLApproxComplex &epsilon(PrecisionT eps) {
        epsilon_ = eps;
        return *this;
    }
    PLApproxComplex &margin(PrecisionT m) {
        margin_ = m;
        return *this;
    }
};

template <class T>
bool operator==(const std::complex<T> &lhs, const PLApproxComplex<T> &rhs) {
    return rhs.compare(lhs);
}
template <class T>
bool operator!=(const std::complex<T> &lhs, const PLApproxComplex<T> &rhs) {
    return !rhs.compare(lhs);
}

template <typename T> PLApproxComplex<T> approx(const std::complex<T> &val) {
    return PLApproxComplex<T>{val};
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const PLApproxComplex<T> &approx) {
    os << approx.describe();
    return os;
}

/**
 * @brief Utility function to compare complex statevector data.
 *
 * @tparam Data_t Floating point data-type.
 * @param data1 StateVector data 1.
 * @param data2 StateVector data 2.
 * @return true Data are approximately equal.
 * @return false Data are not approximately equal.
 */
template <class Data_t, class AllocA, class AllocB>
inline bool
isApproxEqual(const std::vector<Data_t, AllocA> &data1,
              const std::vector<Data_t, AllocB> &data2,
              const typename Data_t::value_type eps =
                  std::numeric_limits<typename Data_t::value_type>::epsilon() *
                  100) {
    return data1 == PLApprox<Data_t, AllocB>(data2).epsilon(eps);
}

/**
 * @brief Utility function to compare complex statevector data.
 *
 * @tparam Data_t Floating point data-type.
 * @param data1 StateVector data 1.
 * @param data2 StateVector data 2.
 * @return true Data are approximately equal.
 * @return false Data are not approximately equal.
 */
template <class Data_t>
inline bool
isApproxEqual(const Data_t &data1, const Data_t &data2,
              typename Data_t::value_type eps =
                  std::numeric_limits<typename Data_t::value_type>::epsilon() *
                  100) {
    return !(data1.real() != Approx(data2.real()).epsilon(eps) ||
             data1.imag() != Approx(data2.imag()).epsilon(eps));
}

#define PL_REQUIRE_THROWS_MATCHES(expr, type, message_match)                   \
    REQUIRE_THROWS_AS(expr, type);                                             \
    REQUIRE_THROWS_WITH(expr, Catch::Matchers::Contains(message_match));
#define PL_CHECK_THROWS_MATCHES(expr, type, message_match)                     \
    CHECK_THROWS_AS(expr, type);                                               \
    CHECK_THROWS_WITH(expr, Catch::Matchers::Contains(message_match));

} // namespace Pennylane::Util
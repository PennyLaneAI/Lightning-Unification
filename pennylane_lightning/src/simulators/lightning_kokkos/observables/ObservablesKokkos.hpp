#pragma once

#include <vector>

#include <Kokkos_Core.hpp>

#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "Error.hpp"
#include "LinearAlgebraKokkos.hpp"
#include "Observables.hpp"
#include "StateVectorKokkos.hpp"
#include "Util.hpp"

// using namespace Pennylane;
/// @cond DEV
namespace {
using namespace Pennylane::Util;
using namespace Pennylane::Observables;
using Pennylane::Lightning_Kokkos::StateVectorKokkos;
} // namespace
/// @endcond

namespace Pennylane::Lightning_Kokkos::Observables {

/**
 * @brief A base class for all observable classes.
 *
 * We note that all subclasses must be immutable (does not provide any setter).
 *
 * @tparam T Floating point type
 */

template <typename T>
class ObservableKokkos
    : public std::enable_shared_from_this<ObservableKokkos<T>> {
  private:
    /**
     * @brief Polymorphic function comparing this to another Observable
     * object.
     *
     * @param Another instance of subclass of Observable<T> to compare
     */
    [[nodiscard]] virtual bool
    isEqual(const ObservableKokkos<T> &other) const = 0;

  protected:
    ObservableKokkos() = default;
    ObservableKokkos(const ObservableKokkos &) = default;
    ObservableKokkos(ObservableKokkos &&) noexcept = default;
    ObservableKokkos &operator=(const ObservableKokkos &) = default;
    ObservableKokkos &operator=(ObservableKokkos &&) noexcept = default;

  public:
    virtual ~ObservableKokkos() = default;

    /**
     * @brief Apply the observable to the given statevector in place.
     */
    virtual void applyInPlace(StateVectorKokkos<T> &sv) const = 0;

    /**
     * @brief Get the name of the observable
     */
    [[nodiscard]] virtual auto getObsName() const -> std::string = 0;

    /**
     * @brief Get the wires the observable applies to.
     */
    [[nodiscard]] virtual auto getWires() const -> std::vector<size_t> = 0;

    /**
     * @brief Test whether this object is equal to another object
     */
    [[nodiscard]] bool operator==(const ObservableKokkos<T> &other) const {
        return typeid(*this) == typeid(other) && isEqual(other);
    }

    /**
     * @brief Test whether this object is different from another object.
     */
    [[nodiscard]] bool operator!=(const ObservableKokkos<T> &other) const {
        return !(*this == other);
    }
};

/**
 * @brief Final class for named observables (PauliX, PauliY, PauliZ, etc.)
 *
 * @tparam StateVectorT State vector class.
 */
template <class StateVectorT>
class NamedObs final : public NamedObsBase<StateVectorT> {
  private:
    using BaseType = NamedObsBase<StateVectorT>;

  public:
    using PrecisionT = typename StateVectorT::PrecisionT;
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
        using Pennylane::Lightning_Kokkos::Gates::Constant::gate_names;
        using Pennylane::Lightning_Kokkos::Gates::Constant::gate_num_params;
        using Pennylane::Lightning_Kokkos::Gates::Constant::gate_wires;
        using Pennylane::Util::lookup;
        using Pennylane::Util::reverse_pairs;

        const auto gate_op = lookup(reverse_pairs(gate_names),
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
  private:
    using BaseType = HermitianObsBase<StateVectorT>;

  public:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using MatrixT = std::vector<ComplexT>;

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
  private:
    using BaseType = TensorProdObsBase<StateVectorT>;

  public:
    using PrecisionT = typename StateVectorT::PrecisionT;

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

/**
 * @brief Final class for a general Hamiltonian representation as a sum of
 * observables.
 *
 * @tparam StateVectorT State vector class.
 */
template <class StateVectorT>
class Hamiltonian final : public HamiltonianBase<StateVectorT> {
  private:
    using BaseType = HamiltonianBase<StateVectorT>;

  public:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

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

    /**
     * @brief Updates the statevector sv:->sv'.
     * @param sv The statevector to update
     */
    void applyInPlace(StateVectorT &sv) const override {

        StateVectorT buffer(sv.getNumQubits());
        buffer.initZeros();

        for (size_t term_idx = 0; term_idx < this->coeffs_.size(); term_idx++) {
            StateVectorT tmp(sv);
            this->obs_[term_idx]->applyInPlace(tmp);
            Lightning_Kokkos::Util::axpy_Kokkos<PrecisionT>(
                ComplexT{this->coeffs_[term_idx], 0.0}, tmp.getData(),
                buffer.getData(), tmp.getLength());
        }
        sv.updateData(buffer);
    }
};

/**
 * @brief Sparse representation of HamiltonianKokkos<T>
 *
 * @tparam T Floating-point precision.
 */
template <typename T>
class SparseHamiltonianKokkos final : public ObservableKokkos<T> {
  public:
    using PrecisionT = T;

  private:
    std::vector<std::complex<T>> data_;
    std::vector<std::size_t> indices_; // colum indices
    std::vector<std::size_t> indptr_;  // row_map
    std::vector<std::size_t> wires_;

    [[nodiscard]] bool
    isEqual(const ObservableKokkos<T> &other) const override {
        const auto &other_cast =
            static_cast<const SparseHamiltonianKokkos<T> &>(other);

        if (data_ != other_cast.data_ || indices_ != other_cast.indices_ ||
            indptr_ != other_cast.indptr_) {
            return false;
        }

        return true;
    }

  public:
    /**
     * @brief Create a SparseHamiltonian from data, indices and indptr in CSR
     * format.
     * @tparam T1 Complex floating point type
     * @tparam T2 std::vector<std::size_t> type
     * @tparam T3 std::vector<std::size_t> type
     * @tparam T4 std::vector<std::size_t> type
     * @param data_arg Arguments to construct data
     * @param indices_arg Arguments to construct indices
     * @param indptr_arg Arguments to construct indptr
     * @param wires_arg Arguments to construct wires
     */
    template <typename T1, typename T2, typename T3 = T2,
              typename T4 = std::vector<std::size_t>>
    SparseHamiltonianKokkos(T1 &&data_arg, T2 &&indices_arg, T3 &&indptr_arg,
                            T4 &&wires_arg)
        : data_{std::forward<T1>(data_arg)}, indices_{std::forward<T2>(
                                                 indices_arg)},
          indptr_{std::forward<T3>(indptr_arg)}, wires_{std::forward<T4>(
                                                     wires_arg)} {
        PL_ASSERT(data_.size() == indices_.size());
    }

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param data_arg Argument to construct data
     * @param indices_arg Argument to construct indices
     * @param indptr_arg Argument to construct ofsets
     * @param wires_arg Argument to construct wires
     */
    static auto create(std::initializer_list<T> data_arg,
                       std::initializer_list<std::size_t> indices_arg,
                       std::initializer_list<std::size_t> indptr_arg,
                       std::initializer_list<std::size_t> wires_arg)
        -> std::shared_ptr<SparseHamiltonianKokkos<T>> {
        return std::shared_ptr<SparseHamiltonianKokkos<T>>(
            new SparseHamiltonianKokkos<T>{
                std::move(data_arg), std::move(indices_arg),
                std::move(indptr_arg), std::move(wires_arg)});
    }

    /**
     * @brief Updates the statevector SV:->SV', where SV' = a*H*SV, and where H
     * is a sparse Hamiltonian.
     */
    void applyInPlace(StateVectorKokkos<T> &sv) const override {
        PL_ABORT_IF_NOT(wires_.size() == sv.getNumQubits(),
                        "SparseH wire count does not match state-vector size");

        StateVectorKokkos<T> d_sv_prime(sv.getNumQubits());

        Lightning_Kokkos::Util::SparseMV_Kokkos<T>(
            sv.getData(), d_sv_prime.getData(), data_, indices_, indptr_);

        sv.updateData(d_sv_prime);
    }

    [[nodiscard]] auto getObsName() const -> std::string override {
        std::ostringstream ss;
        ss << "SparseHamiltonian: {\n'data' : ";
        for (const auto &d : data_)
            ss << d;
        ss << ",\n'indices' : ";
        for (const auto &i : indices_)
            ss << i;
        ss << ",\n'indptr' : ";
        for (const auto &o : indptr_)
            ss << o;
        ss << "\n}";
        return ss.str();
    }
    /**
     * @brief Get the wires the observable applies to.
     */
    [[nodiscard]] auto getWires() const -> std::vector<size_t> override {
        return wires_;
    };
};

} // namespace Pennylane::Lightning_Kokkos::Observables

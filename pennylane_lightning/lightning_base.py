# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
This module contains the base class for all PennyLane Lightning simulator devices, and interfaces with C++ for improved performance.
"""

from typing import List

# from warnings import warn
# from os import getenv
from itertools import islice, product

import numpy as np
from pennylane import (
    QubitDevice,
    math,
    BasisState,
    QubitStateVector,
    Projector,
    Rot,
    QuantumFunctionError,
    DeviceError,
)
from pennylane.devices import DefaultQubit
from pennylane.operation import Tensor, Operation
from pennylane.measurements import MeasurementProcess, Expectation, State
from pennylane.wires import Wires

# tolerance for numerical errors
tolerance = 1e-10

import pennylane as qml

from ._version import __version__

try:
    from .pennylane_lightning_ops import (
        # adjoint_diff,
        # MeasuresC64,
        StateVectorC64,
        # MeasuresC128,
        StateVectorC128,
        backend_info,
    )

    from ._serialize import _serialize_ob, _serialize_observables, _serialize_ops

    CPP_BINARY_AVAILABLE = True
except ModuleNotFoundError:
    CPP_BINARY_AVAILABLE = False

    def backend_info():
        return {"NAME": "NONE"}


if CPP_BINARY_AVAILABLE:

    class LightningBase(QubitDevice):
        """PennyLane Lightning Base device.

        This intermediate base class provides device-agnostic functionality.

        Use of this device requires pre-built binaries or compilation from source. Check out the
        :doc:`/installation` guide for more details.

        Args:
            wires (int): the number of wires to initialize the device with
            c_dtype: Datatypes for statevector representation. Must be one of ``np.complex64`` or ``np.complex128``.
            shots (int): How many times the circuit should be evaluated (or sampled) to estimate
                the expectation values. Defaults to ``None`` if not specified. Setting
                to ``None`` results in computing statistics like expectation values and
                variances analytically.
            batch_obs (bool): Determine whether we process observables in parallel when computing the
                jacobian. This value is only relevant when the lightning qubit is built with OpenMP.
        """

        pennylane_requires = ">=0.30"
        version = __version__
        author = "Xanadu Inc."
        _CPP_BINARY_AVAILABLE = True

        def __init__(
            self,
            wires,
            *,
            c_dtype=np.complex128,
            shots=None,
            batch_obs=False,
            analytic=None,
        ):
            if c_dtype is np.complex64:
                r_dtype = np.float32
                self.use_csingle = True
            elif c_dtype is np.complex128:
                r_dtype = np.float64
                self.use_csingle = False
            else:
                raise TypeError(f"Unsupported complex Type: {c_dtype}")
            super().__init__(
                wires, shots=shots, r_dtype=r_dtype, c_dtype=c_dtype, analytic=analytic
            )
            self._batch_obs = batch_obs

        @property
        def stopping_condition(self):
            """.BooleanFn: Returns the stopping condition for the device. The returned
            function accepts a queuable object (including a PennyLane operation
            and observable) and returns ``True`` if supported by the device."""

            def accepts_obj(obj):
                if obj.name == "QFT" and len(obj.wires) < 10:
                    return True
                if obj.name == "GroverOperator" and len(obj.wires) < 13:
                    return True
                return (not isinstance(obj, qml.tape.QuantumTape)) and getattr(
                    self, "supports_operation", lambda name: False
                )(obj.name)

            return qml.BooleanFn(accepts_obj)

        @classmethod
        def capabilities(cls):
            capabilities = super().capabilities().copy()
            capabilities.update(
                model="qubit",
                supports_analytic_computation=True,
                supports_broadcasting=False,
                returns_state=True,
            )
            return capabilities

        # To be able to validate the adjoint method [_validate_adjoint_method(device)],
        #  the qnode requires the definition of:
        # ["_apply_operation", "_apply_unitary", "adjoint_jacobian"]
        def _apply_operation():
            pass

        def _apply_unitary():
            pass

        # @staticmethod
        # def _check_adjdiff_supported_measurements(measurements: List[MeasurementProcess]):
        #     """Check whether given list of measurement is supported by adjoint_diff.

        #     Args:
        #         measurements (List[MeasurementProcess]): a list of measurement processes to check.

        #     Returns:
        #         Expectation or State: a common return type of measurements.
        #     """
        #     if len(measurements) == 0:
        #         return None

        #     if len(measurements) == 1 and measurements[0].return_type is State:
        #         return State

        #     # Now the return_type of measurement processes must be expectation
        #     if not all([m.return_type is Expectation for m in measurements]):
        #         raise QuantumFunctionError(
        #             "Adjoint differentiation method does not support expectation return type "
        #             "mixed with other return types"
        #         )

        #     for m in measurements:
        #         if not isinstance(m.obs, Tensor):
        #             if isinstance(m.obs, Projector):
        #                 raise QuantumFunctionError(
        #                     "Adjoint differentiation method does not support the Projector observable"
        #                 )
        #         else:
        #             if any([isinstance(o, Projector) for o in m.obs.non_identity_obs]):
        #                 raise QuantumFunctionError(
        #                     "Adjoint differentiation method does not support the Projector observable"
        #                 )
        #     return Expectation

        # @staticmethod
        # def _check_adjdiff_supported_operations(operations):
        #     """Check Lightning adjoint differentiation method support for a tape.

        #     Raise ``QuantumFunctionError`` if ``tape`` contains not supported measurements,
        #     observables, or operations by the Lightning adjoint differentiation method.

        #     Args:
        #         tape (.QuantumTape): quantum tape to differentiate.
        #     """
        #     for op in operations:
        #         if op.num_params > 1 and not isinstance(op, Rot):
        #             raise QuantumFunctionError(
        #                 f"The {op.name} operation is not supported using "
        #                 'the "adjoint" differentiation method'
        #             )

        # def _process_jacobian_tape(self, tape, starting_state, use_device_state):
        #     # To support np.complex64 based on the type of self._state
        #     if self.use_csingle:
        #         create_ops_list = adjoint_diff.create_ops_list_C64
        #     else:
        #         create_ops_list = adjoint_diff.create_ops_list_C128

        #     # Initialization of state
        #     if starting_state is not None:
        #         if starting_state.size != 2 ** len(self.wires):
        #             raise QuantumFunctionError(
        #                 "The number of qubits of starting_state must be the same as "
        #                 "that of the device."
        #             )
        #         ket = self._asarray(starting_state, dtype=self.C_DTYPE)
        #     else:
        #         if not use_device_state:
        #             self.reset()
        #             self.apply(tape.operations)
        #         ket = self._pre_rotated_state

        #     obs_serialized = _serialize_observables(tape, self.wire_map, use_csingle=self.use_csingle)
        #     ops_serialized, use_sp = _serialize_ops(tape, self.wire_map)

        #     ops_serialized = create_ops_list(*ops_serialized)

        #     # We need to filter out indices in trainable_params which do not
        #     # correspond to operators.
        #     trainable_params = sorted(tape.trainable_params)
        #     if len(trainable_params) == 0:
        #         return None

        #     tp_shift = []
        #     record_tp_rows = []
        #     all_params = 0

        #     for op_idx, tp in enumerate(trainable_params):
        #         # get op_idx-th operator among differentiable operators
        #         op, _, _ = tape.get_operation(op_idx)
        #         if isinstance(op, Operation) and not isinstance(op, (BasisState, QubitStateVector)):
        #             # We now just ignore non-op or state preps
        #             tp_shift.append(tp)
        #             record_tp_rows.append(all_params)
        #         all_params += 1

        #     if use_sp:
        #         # When the first element of the tape is state preparation. Still, I am not sure
        #         # whether there must be only one state preparation...
        #         tp_shift = [i - 1 for i in tp_shift]

        #     ket = ket.reshape(-1)
        #     state_vector = StateVectorC64(ket) if self.use_csingle else StateVectorC128(ket)
        #     return {
        #         "state_vector": state_vector,
        #         "obs_serialized": obs_serialized,
        #         "ops_serialized": ops_serialized,
        #         "tp_shift": tp_shift,
        #         "record_tp_rows": record_tp_rows,
        #         "all_params": all_params,
        #     }

        # def adjoint_jacobian(self, tape, starting_state=None, use_device_state=False):
        #     if self.shots is not None:
        #         warn(
        #             "Requested adjoint differentiation to be computed with finite shots."
        #             " The derivative is always exact when using the adjoint differentiation method.",
        #             UserWarning,
        #         )

        #     tape_return_type = self._check_adjdiff_supported_measurements(tape.measurements)

        #     if not tape_return_type:  # the tape does not have measurements
        #         return np.array([], dtype=self._state.dtype)

        #     if tape_return_type is State:
        #         raise QuantumFunctionError(
        #             "This method does not support statevector return type. "
        #             "Use vjp method instead for this purpose."
        #         )

        #     self._check_adjdiff_supported_operations(tape.operations)

        #     processed_data = self._process_jacobian_tape(tape, starting_state, use_device_state)

        #     if not processed_data:  # training_params is empty
        #         return np.array([], dtype=self._state.dtype)

        #     trainable_params = processed_data["tp_shift"]

        #     # If requested batching over observables, chunk into OMP_NUM_THREADS sized chunks.
        #     # This will allow use of Lightning with adjoint for large-qubit numbers AND large
        #     # numbers of observables, enabling choice between compute time and memory use.
        #     requested_threads = int(getenv("OMP_NUM_THREADS", "1"))

        #     if self._batch_obs and requested_threads > 1:
        #         obs_partitions = _chunk_iterable(processed_data["obs_serialized"], requested_threads)
        #         jac = []
        #         for obs_chunk in obs_partitions:
        #             jac_local = adjoint_diff.adjoint_jacobian(
        #                 processed_data["state_vector"],
        #                 obs_chunk,
        #                 processed_data["ops_serialized"],
        #                 trainable_params,
        #             )
        #             jac.extend(jac_local)
        #     else:
        #         jac = adjoint_diff.adjoint_jacobian(
        #             processed_data["state_vector"],
        #             processed_data["obs_serialized"],
        #             processed_data["ops_serialized"],
        #             trainable_params,
        #         )
        #     jac = np.array(jac)
        #     jac = jac.reshape(-1, len(trainable_params))
        #     jac_r = np.zeros((jac.shape[0], processed_data["all_params"]))
        #     jac_r[:, processed_data["record_tp_rows"]] = jac
        #     return self._adjoint_jacobian_processing(jac_r) if qml.active_return() else jac_r

        # @staticmethod
        # def _adjoint_jacobian_processing(jac):
        #     """
        #     Post-process the Jacobian matrix returned by ``adjoint_jacobian`` for
        #     the new return type system.
        #     """
        #     jac = np.squeeze(jac)

        #     if jac.ndim == 0:
        #         return np.array(jac)

        #     if jac.ndim == 1:
        #         return tuple(np.array(j) for j in jac)

        #     # must be 2-dimensional
        #     return tuple(tuple(np.array(j_) for j_ in j) for j in jac)

        # def vjp(self, measurements, dy, starting_state=None, use_device_state=False):
        #     """Generate the processing function required to compute the vector-Jacobian products of a tape.

        #     This function can be used with multiple expectation values or a quantum state. When a quantum state
        #     is given,

        #     .. code-block:: python

        #         vjp_f = dev.vjp([qml.state()], dy)
        #         vjp = vjp_f(tape)

        #     computes :math:`w = (w_1,\\cdots,w_m)` where

        #     .. math::

        #         w_k = \\langle v| \\frac{\\partial}{\\partial \\theta_k} | \\psi_{\\pmb{\\theta}} \\rangle.

        #     Here, :math:`m` is the total number of trainable parameters, :math:`\\pmb{\\theta}` is the vector of trainable parameters and :math:`\\psi_{\\pmb{\\theta}}`
        #     is the output quantum state.

        #     Args:
        #         measurements (list): List of measurement processes for vector-Jacobian product. Now it must be expectation values or a quantum state.
        #         dy (tensor_like): Gradient-output vector. Must have shape matching the output shape of the corresponding tape, i.e. number of measurements if the return type is expectation or :math:`2^N` if the return type is statevector
        #         starting_state (tensor_like): post-forward pass state to start execution with. It should be
        #             complex-valued. Takes precedence over ``use_device_state``.
        #         use_device_state (bool): use current device state to initialize. A forward pass of the same
        #             circuit should be the last thing the device has executed. If a ``starting_state`` is
        #             provided, that takes precedence.

        #     Returns:
        #         The processing function required to compute the vector-Jacobian products of a tape.
        #     """
        #     if self.shots is not None:
        #         warn(
        #             "Requested adjoint differentiation to be computed with finite shots."
        #             " The derivative is always exact when using the adjoint differentiation method.",
        #             UserWarning,
        #         )

        #     tape_return_type = self._check_adjdiff_supported_measurements(measurements)

        #     if math.allclose(dy, 0) or tape_return_type is None:
        #         return lambda tape: math.convert_like(np.zeros(len(tape.trainable_params)), dy)

        #     if tape_return_type is Expectation:
        #         if len(dy) != len(measurements):
        #             raise ValueError(
        #                 "Number of observables in the tape must be the same as the length of dy in the vjp method"
        #             )

        #         if np.iscomplexobj(dy):
        #             raise ValueError(
        #                 "The vjp method only works with a real-valued dy when the tape is returning an expectation value"
        #             )

        #         ham = qml.Hamiltonian(dy, [m.obs for m in measurements])

        #         def processing_fn(tape):
        #             nonlocal ham
        #             num_params = len(tape.trainable_params)

        #             if num_params == 0:
        #                 return np.array([], dtype=self._state.dtype)

        #             new_tape = tape.copy()
        #             new_tape._measurements = [qml.expval(ham)]

        #             return self.adjoint_jacobian(new_tape, starting_state, use_device_state)

        #         return processing_fn

        #     if tape_return_type is State:
        #         if len(dy) != 2 ** len(self.wires):
        #             raise ValueError(
        #                 "Size of the provided vector dy must be the same as the size of the statevector"
        #             )
        #         if np.isrealobj(dy):
        #             warn(
        #                 "The vjp method only works with complex-valued dy when the tape is returning a statevector. Upcasting dy."
        #             )

        #         dy = dy.astype(self.C_DTYPE)

        #         def processing_fn(tape):
        #             nonlocal dy
        #             processed_data = self._process_jacobian_tape(tape, starting_state, use_device_state)
        #             return adjoint_diff.statevector_vjp(
        #                 processed_data["state_vector"],
        #                 processed_data["ops_serialized"],
        #                 dy,
        #                 processed_data["tp_shift"],
        #             )

        #         return processing_fn

        # def batch_vjp(
        #     self, tapes, dys, reduction="append", starting_state=None, use_device_state=False
        # ):
        #     """Generate the processing function required to compute the vector-Jacobian products
        #     of a batch of tapes.

        #     Args:
        #         tapes (Sequence[.QuantumTape]): sequence of quantum tapes to differentiate
        #         dys (Sequence[tensor_like]): Sequence of gradient-output vectors ``dy``. Must be the
        #             same length as ``tapes``. Each ``dy`` tensor should have shape
        #             matching the output shape of the corresponding tape.
        #         reduction (str): Determines how the vector-Jacobian products are returned.
        #             If ``append``, then the output of the function will be of the form
        #             ``List[tensor_like]``, with each element corresponding to the VJP of each
        #             input tape. If ``extend``, then the output VJPs will be concatenated.
        #         starting_state (tensor_like): post-forward pass state to start execution with. It should be
        #             complex-valued. Takes precedence over ``use_device_state``.
        #         use_device_state (bool): use current device state to initialize. A forward pass of the same
        #             circuit should be the last thing the device has executed. If a ``starting_state`` is
        #             provided, that takes precedence.

        #     Returns:
        #         The processing function required to compute the vector-Jacobian products of a batch of tapes.
        #     """
        #     fns = []

        #     # Loop through the tapes and dys vector
        #     for tape, dy in zip(tapes, dys):
        #         fn = self.vjp(
        #             tape.measurements,
        #             dy,
        #             starting_state=starting_state,
        #             use_device_state=use_device_state,
        #         )
        #         fns.append(fn)

        #     def processing_fns(tapes):
        #         vjps = []
        #         for t, f in zip(tapes, fns):
        #             vjp = f(t)

        #             # make sure vjp is iterable if using extend reduction
        #             if (
        #                 not isinstance(vjp, tuple)
        #                 and getattr(reduction, "__name__", reduction) == "extend"
        #             ):
        #                 vjp = (vjp,)

        #             if isinstance(reduction, str):
        #                 getattr(vjps, reduction)(vjp)
        #             elif callable(reduction):
        #                 reduction(vjps, vjp)

        #         return vjps

        #     return processing_fns

        # def probability(self, wires=None, shot_range=None, bin_size=None):
        #     """Return the probability of each computational basis state.

        #     Devices that require a finite number of shots always return the
        #     estimated probability.

        #     Args:
        #         wires (Iterable[Number, str], Number, str, Wires): wires to return
        #             marginal probabilities for. Wires not provided are traced out of the system.
        #         shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
        #             to use. If not specified, all samples are used.
        #         bin_size (int): Divides the shot range into bins of size ``bin_size``, and
        #             returns the measurement statistic separately over each bin. If not
        #             provided, the entire shot range is treated as a single bin.

        #     Returns:
        #         array[float]: list of the probabilities
        #     """
        #     if self.shots is not None:
        #         return self.estimate_probability(wires=wires, shot_range=shot_range, bin_size=bin_size)

        #     wires = wires or self.wires
        #     wires = Wires(wires)

        #     # translate to wire labels used by device
        #     device_wires = self.map_wires(wires)

        #     if (
        #         device_wires
        #         and len(device_wires) > 1
        #         and (not np.all(np.array(device_wires)[:-1] <= np.array(device_wires)[1:]))
        #     ):
        #         raise RuntimeError(
        #             "Lightning does not currently support out-of-order indices for probabilities"
        #         )

        #     # To support np.complex64 based on the type of self._state
        #     dtype = self._state.dtype
        #     ket = np.ravel(self._state)

        #     state_vector = StateVectorC64(ket) if self.use_csingle else StateVectorC128(ket)
        #     M = MeasuresC64(state_vector) if self.use_csingle else MeasuresC128(state_vector)
        #     return M.probs(device_wires)

        # def generate_samples(self):
        #     """Generate samples

        #     Returns:
        #         array[int]: array of samples in binary representation with shape ``(dev.shots, dev.num_wires)``
        #     """
        #     # Initialization of state
        #     ket = np.ravel(self._state)

        #     state_vector = StateVectorC64(ket) if self.use_csingle else StateVectorC128(ket)
        #     M = MeasuresC64(state_vector) if self.use_csingle else MeasuresC128(state_vector)
        #     if self._mcmc:
        #         return M.generate_mcmc_samples(
        #             len(self.wires), self._kernel_name, self._num_burnin, self.shots
        #         ).astype(int, copy=False)
        #     else:
        #         return M.generate_samples(len(self.wires), self.shots).astype(int, copy=False)

        # def expval(self, observable, shot_range=None, bin_size=None):
        #     """Expectation value of the supplied observable.

        #     Args:
        #         observable: A PennyLane observable.
        #         shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
        #             to use. If not specified, all samples are used.
        #         bin_size (int): Divides the shot range into bins of size ``bin_size``, and
        #             returns the measurement statistic separately over each bin. If not
        #             provided, the entire shot range is treated as a single bin.

        #     Returns:
        #         Expectation value of the observable
        #     """
        #     if observable.name in [
        #         "Identity",
        #         "Projector",
        #     ]:
        #         return super().expval(observable, shot_range=shot_range, bin_size=bin_size)

        #     if self.shots is not None:
        #         # estimate the expectation value
        #         # LightningQubit doesn't support sampling yet
        #         samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)
        #         return np.squeeze(np.mean(samples, axis=0))

        #     # Initialization of state
        #     ket = np.ravel(self._pre_rotated_state)

        #     state_vector = StateVectorC64(ket) if self.use_csingle else StateVectorC128(ket)
        #     M = MeasuresC64(state_vector) if self.use_csingle else MeasuresC128(state_vector)
        #     if observable.name == "SparseHamiltonian":
        #         if Kokkos_info()["USE_KOKKOS"] == True:
        #             # ensuring CSR sparse representation.

        #             CSR_SparseHamiltonian = observable.sparse_matrix(wire_order=self.wires).tocsr(
        #                 copy=False
        #             )
        #             return M.expval(
        #                 CSR_SparseHamiltonian.indptr,
        #                 CSR_SparseHamiltonian.indices,
        #                 CSR_SparseHamiltonian.data,
        #             )
        #         raise NotImplementedError(
        #             "The expval of a SparseHamiltonian requires Kokkos and Kokkos Kernels."
        #         )

        #     if (
        #         observable.name in ["Hamiltonian", "Hermitian"]
        #         or (observable.arithmetic_depth > 0)
        #         or isinstance(observable.name, List)
        #     ):
        #         ob_serialized = _serialize_ob(observable, self.wire_map, use_csingle=self.use_csingle)
        #         return M.expval(ob_serialized)

        #     # translate to wire labels used by device
        #     observable_wires = self.map_wires(observable.wires)

        #     return M.expval(observable.name, observable_wires)

        # def var(self, observable, shot_range=None, bin_size=None):
        #     """Variance of the supplied observable.

        #     Args:
        #         observable: A PennyLane observable.
        #         shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
        #             to use. If not specified, all samples are used.
        #         bin_size (int): Divides the shot range into bins of size ``bin_size``, and
        #             returns the measurement statistic separately over each bin. If not
        #             provided, the entire shot range is treated as a single bin.

        #     Returns:
        #         Variance of the observable
        #     """
        #     if observable.name in [
        #         "Identity",
        #         "Projector",
        #     ]:
        #         return super().var(observable, shot_range=shot_range, bin_size=bin_size)

        #     if self.shots is not None:
        #         # estimate the var
        #         # LightningQubit doesn't support sampling yet
        #         samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)
        #         return np.squeeze(np.var(samples, axis=0))

        #     # Initialization of state
        #     ket = np.ravel(self._pre_rotated_state)

        #     state_vector = StateVectorC64(ket) if self.use_csingle else StateVectorC128(ket)
        #     M = MeasuresC64(state_vector) if self.use_csingle else MeasuresC128(state_vector)

        #     if observable.name == "SparseHamiltonian":
        #         if Kokkos_info()["USE_KOKKOS"] == True:
        #             # ensuring CSR sparse representation.

        #             CSR_SparseHamiltonian = observable.sparse_matrix(wire_order=self.wires).tocsr(
        #                 copy=False
        #             )
        #             return M.var(
        #                 CSR_SparseHamiltonian.indptr,
        #                 CSR_SparseHamiltonian.indices,
        #                 CSR_SparseHamiltonian.data,
        #             )
        #         raise NotImplementedError(
        #             "The expval of a SparseHamiltonian requires Kokkos and Kokkos Kernels."
        #         )

        #     if (
        #         observable.name in ["Hamiltonian", "Hermitian"]
        #         or (observable.arithmetic_depth > 0)
        #         or isinstance(observable.name, List)
        #     ):
        #         ob_serialized = _serialize_ob(observable, self.wire_map, use_csingle=self.use_csingle)
        #         return M.var(ob_serialized)

        #     # translate to wire labels used by device
        #     observable_wires = self.map_wires(observable.wires)

        #     return M.var(observable.name, observable_wires)

        # def _get_diagonalizing_gates(self, circuit: qml.tape.QuantumTape) -> List[Operation]:
        #     skip_diagonalizing = lambda obs: isinstance(obs, qml.Hamiltonian) or (
        #         isinstance(obs, qml.ops.Sum) and obs._pauli_rep is not None
        #     )
        #     meas_filtered = list(
        #         filter(lambda m: m.obs is None or not skip_diagonalizing(m.obs), circuit.measurements)
        #     )
        #     return super()._get_diagonalizing_gates(qml.tape.QuantumScript(measurements=meas_filtered))

else:  # binaries not available

    class LightningBase(DefaultQubit):  # pragma: no cover
        pennylane_requires = ">=0.30"
        version = __version__
        author = "Xanadu Inc."
        _CPP_BINARY_AVAILABLE = False

        def __init__(self, wires, *, c_dtype=np.complex128, **kwargs):
            if c_dtype is np.complex64:
                r_dtype = np.float32
            elif c_dtype is np.complex128:
                r_dtype = np.float64
            else:
                raise TypeError(f"Unsupported complex Type: {c_dtype}")
            super().__init__(wires, r_dtype=r_dtype, c_dtype=c_dtype, **kwargs)

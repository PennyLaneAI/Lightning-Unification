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
This module contains the :class:`~.LightningQubit` class, a PennyLane simulator device that
interfaces with C++ for fast linear algebra calculations.
"""
import numpy as np
from warnings import warn

from .lightning_base import backend_info, LightningBase

if backend_info()["NAME"] == "lightning.kokkos":
    from typing import List
    from itertools import islice
    from os import getenv

    from .pennylane_lightning_ops import (
        allocate_aligned_array,
        get_alignment,
        best_alignment,
    )

    from pennylane import (
        math,
        BasisState,
        QubitStateVector,
        Projector,
        Rot,
        DeviceError,
        QuantumFunctionError,
    )
    from pennylane.operation import Tensor, Operation
    from pennylane.ops.op_math import Adjoint
    from pennylane.measurements import MeasurementProcess, Expectation, State
    from pennylane.wires import Wires

    import pennylane as qml

    from ._version import __version__

    from .pennylane_lightning_ops import (
        InitializationSettings,
        print_configuration,
        MeasurementsC64,
        StateVectorC64,
        MeasurementsC128,
        StateVectorC128,
    )

    from .pennylane_lightning_ops.algorithms import (
        AdjointJacobianC64,
        create_ops_listC64,
        AdjointJacobianC128,
        create_ops_listC128,
    )

    from ._serialize import _serialize_ob, _serialize_observables, _serialize_ops

    def _kokkos_dtype(dtype):
        if dtype not in [np.complex128, np.complex64]:
            raise ValueError(f"Data type is not supported for state-vector computation: {dtype}")
        return StateVectorC128 if dtype == np.complex128 else StateVectorC64

    def _kokkos_configuration():
        return print_configuration()

    def _chunk_iterable(it, num_chunks):
        "Lazy-evaluated chunking of given iterable from https://stackoverflow.com/a/22045226"
        it = iter(it)
        return iter(lambda: tuple(islice(it, num_chunks)), ())

    allowed_operations = {
        "Identity",
        "BasisState",
        "QubitStateVector",
        "QubitUnitary",
        "ControlledQubitUnitary",
        "MultiControlledX",
        "DiagonalQubitUnitary",
        "PauliX",
        "PauliY",
        "PauliZ",
        "MultiRZ",
        "Hadamard",
        "S",
        "Adjoint(S)",
        "T",
        "Adjoint(T)",
        "SX",
        "Adjoint(SX)",
        "CNOT",
        "SWAP",
        "ISWAP",
        "PSWAP",
        "Adjoint(ISWAP)",
        "SISWAP",
        "Adjoint(SISWAP)",
        "SQISW",
        "CSWAP",
        "Toffoli",
        "CY",
        "CZ",
        "PhaseShift",
        "ControlledPhaseShift",
        "CPhase",
        "RX",
        "RY",
        "RZ",
        "Rot",
        "CRX",
        "CRY",
        "CRZ",
        "CRot",
        "IsingXX",
        "IsingYY",
        "IsingZZ",
        "IsingXY",
        "SingleExcitation",
        "SingleExcitationPlus",
        "SingleExcitationMinus",
        "DoubleExcitation",
        "DoubleExcitationPlus",
        "DoubleExcitationMinus",
        "QubitCarry",
        "QubitSum",
        "OrbitalRotation",
        "QFT",
        "ECR",
    }

    allowed_observables = {
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "Hermitian",
        "Identity",
        "SparseHamiltonian",
        "Hamiltonian",
        "Sum",
        "SProd",
        "Prod",
        "Exp",
    }

    class LightningKokkos(LightningBase):
        """PennyLane Lightning Kokkos device.

        A device that interfaces with C++ to perform fast linear algebra calculations.

        Use of this device requires pre-built binaries or compilation from source. Check out the
        :doc:`/installation` guide for more details.

        Args:
            wires (int): the number of wires to initialize the device with
            sync (bool): immediately sync with host-sv after applying operations
            c_dtype: Datatypes for statevector representation. Must be one of ``np.complex64`` or ``np.complex128``.
            kokkos_args (InitializationSettings): binding for Kokkos::InitializationSettings (threading parameters).
            shots (int): How many times the circuit should be evaluated (or sampled) to estimate
                the expectation values. Defaults to ``None`` if not specified. Setting
                to ``None`` results in computing statistics like expectation values and
                variances analytically.
        """

        name = "Lightning Kokkos PennyLane plugin"
        short_name = "lightning.kokkos"
        kokkos_config = {}
        operations = allowed_operations
        observables = allowed_observables

        def __init__(
            self,
            wires,
            *,
            sync=True,
            c_dtype=np.complex128,
            shots=None,
            batch_obs=False,
            kokkos_args=None,
        ):
            super().__init__(wires, shots=shots, c_dtype=c_dtype)

            if kokkos_args is None:
                self._kokkos_state = _kokkos_dtype(c_dtype)(self.num_wires)
            elif isinstance(kokkos_args, InitializationSettings):
                self._kokkos_state = _kokkos_dtype(c_dtype)(self.num_wires, kokkos_args)
            else:
                raise TypeError(
                    f"Argument kokkos_args must be of {type(InitializationSettings())} but it is of {type(kokkos_args)}."
                )
            self._sync = sync

            if not LightningKokkos.kokkos_config:
                LightningKokkos.kokkos_config = _kokkos_configuration()

            # Create the initial state. Internally, we store the
            # state as an array of dimension [2]*wires.
            self._pre_rotated_state = _kokkos_dtype(c_dtype)(self.num_wires)

        @staticmethod
        def _asarray(arr, dtype=None):
            arr = np.asarray(arr)  # arr is not copied

            if arr.dtype.kind not in ["f", "c"]:
                return arr

            if not dtype:
                dtype = arr.dtype

            # We allocate a new aligned memory and copy data to there if alignment or dtype mismatches
            # Note that get_alignment does not necessarily return CPUMemoryModel(Unaligned) even for
            # numpy allocated memory as the memory location happens to be aligned.
            if int(get_alignment(arr)) < int(best_alignment()) or arr.dtype != dtype:
                new_arr = allocate_aligned_array(arr.size, np.dtype(dtype)).reshape(arr.shape)
                np.copyto(new_arr, arr)
                arr = new_arr
            return arr

        def _create_basis_state(self, index):
            """Return a computational basis state over all wires.
            Args:
                index (int): integer representing the computational basis state
            Returns:
                array[complex]: complex array of shape ``[2]*self.num_wires``
                representing the statevector of the basis state
            Note: This function does not support broadcasted inputs yet.
            """
            self._kokkos_state.setBasisState(index)

        def reset(self):
            """Reset the device"""
            super().reset()

            # init the state vector to |00..0>
            self._kokkos_state.resetStateVector()  # Sync reset

        def syncH2D(self, state_vector):
            """Copy the state vector data on host provided by the user to the state vector on the device
            Args:
                state_vector(array[complex]): the state vector array on host.
            **Example**
            >>> dev = qml.device('lightning.kokkos', wires=3)
            >>> obs = qml.Identity(0) @ qml.PauliX(1) @ qml.PauliY(2)
            >>> obs1 = qml.Identity(1)
            >>> H = qml.Hamiltonian([1.0, 1.0], [obs1, obs])
            >>> state_vector = np.array([0.0 + 0.0j, 0.0 + 0.1j, 0.1 + 0.1j, 0.1 + 0.2j,
                0.2 + 0.2j, 0.3 + 0.3j, 0.3 + 0.4j, 0.4 + 0.5j,], dtype=np.complex64,)
            >>> dev.syncH2D(state_vector)
            >>> res = dev.expval(H)
            >>> print(res)
            1.0
            """
            self._kokkos_state.HostToDevice(state_vector.ravel(order="C"))

        def syncD2H(self, state_vector):
            """Copy the state vector data on device to a state vector on the host provided by the user
            Args:
                state_vector(array[complex]): the state vector array on host
            **Example**
            >>> dev = qml.device('lightning.kokkos', wires=1)
            >>> dev.apply([qml.PauliX(wires=[0])])
            >>> state_vector = np.zeros(2**dev.num_wires).astype(dev.C_DTYPE)
            >>> dev.syncD2H(state_vector)
            >>> print(state_vector)
            [0.+0.j 1.+0.j]
            """
            self._kokkos_state.DeviceToHost(state_vector.ravel(order="C"))

        @property
        def state(self):
            """Copy the state vector data from the device to the host. A state vector Numpy array is explicitly allocated on the host to store and return the data.
            **Example**
            >>> dev = qml.device('lightning.kokkos', wires=1)
            >>> dev.apply([qml.PauliX(wires=[0])])
            >>> print(dev.state)
            [0.+0.j 1.+0.j]
            """
            state = np.zeros(2**self.num_wires, dtype=self.C_DTYPE)
            state = self._asarray(state, dtype=self.C_DTYPE)
            self.syncD2H(state)
            return state

        @property
        def state_vector(self):
            """Returns a handle to the statevector."""
            return self._kokkos_state

        def _apply_state_vector(self, state, device_wires):
            """Initialize the internal state vector in a specified state.

            Args:
                state (array[complex]): normalized input state of length ``2**len(wires)``
                    or broadcasted state of shape ``(batch_size, 2**len(wires))``
                device_wires (Wires): wires that get initialized in the state
            """

            if isinstance(state, self._kokkos_state.__class__):
                state_data = np.zeros(state.size, dtype=self.C_DTYPE)
                state_data = self._asarray(state_data, dtype=self.C_DTYPE)
                state.DeviceToHost(state_data.ravel(order="C"))
                state = state_data

            ravelled_indices, state = self._preprocess_state_vector(state, device_wires)

            # translate to wire labels used by device
            device_wires = self.map_wires(device_wires)
            output_shape = [2] * self.num_wires

            if len(device_wires) == self.num_wires and Wires(sorted(device_wires)) == device_wires:
                # Initialize the entire device state with the input state
                self.syncH2D(self._reshape(state, output_shape))
                return

            self._kokkos_state.setStateVector(ravelled_indices, state)  # this operation on device

        def _apply_basis_state(self, state, wires):
            """Initialize the state vector in a specified computational basis state.

            Args:
                state (array[int]): computational basis state of shape ``(wires,)``
                    consisting of 0s and 1s.
                wires (Wires): wires that the provided computational state should be initialized on

            Note: This function does not support broadcasted inputs yet.
            """
            num = self._get_basis_state_index(state, wires)
            self._create_basis_state(num)

        def apply_lightning(self, operations):
            """Apply a list of operations to the state tensor.

            Args:
                operations (list[~pennylane.operation.Operation]): operations to apply
                dtype (type): Type of numpy ``complex`` to be used. Can be important
                to specify for large systems for memory allocation purposes.

            Returns:
                array[complex]: the output state tensor
            """
            # Skip over identity operations instead of performing
            # matrix multiplication with the identity.
            invert_param = False
            state = self.state_vector

            for o in operations:
                if str(o.name) == "Identity":
                    continue
                name = o.name
                if isinstance(o, Adjoint):
                    name = o.base.name
                    invert_param = True
                method = getattr(state, name, None)

                wires = self.wires.indices(o.wires)

                if method is None:
                    # Inverse can be set to False since qml.matrix(o) is already in inverted form
                    try:
                        mat = qml.matrix(o)
                    except AttributeError:  # pragma: no cover
                        # To support older versions of PL
                        mat = o.matrix

                    if len(mat) == 0:
                        raise Exception("Unsupported operation")
                    state.apply(
                        name,
                        wires,
                        False,
                        [],
                        mat.ravel(order="C"),  # inv = False: Matrix already in correct form;
                    )  # Parameters can be ignored for explicit matrices; F-order for cuQuantum

                else:
                    param = o.parameters
                    method(wires, invert_param, param)

        def apply(self, operations, rotations=None, **kwargs):
            # State preparation is currently done in Python
            if operations:  # make sure operations[0] exists
                if isinstance(operations[0], QubitStateVector):
                    self._apply_state_vector(
                        operations[0].parameters[0].copy(), operations[0].wires
                    )
                    del operations[0]
                elif isinstance(operations[0], BasisState):
                    self._apply_basis_state(operations[0].parameters[0], operations[0].wires)
                    del operations[0]

            for operation in operations:
                if isinstance(operation, (QubitStateVector, BasisState)):
                    raise DeviceError(
                        f"Operation {operation.name} cannot be used after other Operations have already been applied on a {self.short_name} device."
                    )

            self.apply_lightning(operations)
            # if operations:
            #     self._pre_rotated_state = self.apply_lightning(self._state, operations)
            # else:
            #     self._pre_rotated_state = self._state

            # if rotations:
            #     self._state = self.apply_lightning(np.copy(self._pre_rotated_state), rotations)
            # else:
            #     self._state = self._pre_rotated_state

        def expval(self, observable, shot_range=None, bin_size=None):
            """Expectation value of the supplied observable.

            Args:
                observable: A PennyLane observable.
                shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
                    to use. If not specified, all samples are used.
                bin_size (int): Divides the shot range into bins of size ``bin_size``, and
                    returns the measurement statistic separately over each bin. If not
                    provided, the entire shot range is treated as a single bin.

            Returns:
                Expectation value of the observable
            """
            if observable.name in [
                "Projector",
            ]:
                return super().expval(observable, shot_range=shot_range, bin_size=bin_size)

            if self.shots is not None:
                # estimate the expectation value
                # LightningQubit doesn't support sampling yet
                samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)
                return np.squeeze(np.mean(samples, axis=0))

            # Initialization of state
            M = (
                MeasurementsC64(self.state_vector)
                if self.use_csingle
                else MeasurementsC128(self.state_vector)
            )
            if observable.name == "SparseHamiltonian":
                CSR_SparseHamiltonian = observable.sparse_matrix(wire_order=self.wires).tocsr(
                    copy=False
                )
                return M.expval(
                    CSR_SparseHamiltonian.indptr,
                    CSR_SparseHamiltonian.indices,
                    CSR_SparseHamiltonian.data,
                )

            if (
                observable.name in ["Hamiltonian", "Hermitian"]
                or (observable.arithmetic_depth > 0)
                or isinstance(observable.name, List)
            ):
                ob_serialized = _serialize_ob(
                    observable, self.wire_map, use_csingle=self.use_csingle
                )
                return M.expval(ob_serialized)

            # translate to wire labels used by device
            observable_wires = self.map_wires(observable.wires)

            return M.expval(observable.name, observable_wires)

        def var(self, observable, shot_range=None, bin_size=None):
            """Variance of the supplied observable.

            Args:
                observable: A PennyLane observable.
                shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
                    to use. If not specified, all samples are used.
                bin_size (int): Divides the shot range into bins of size ``bin_size``, and
                    returns the measurement statistic separately over each bin. If not
                    provided, the entire shot range is treated as a single bin.

            Returns:
                Variance of the observable
            """
            if observable.name in [
                "Projector",
            ]:
                return super().var(observable, shot_range=shot_range, bin_size=bin_size)

            if self.shots is not None:
                # estimate the var
                # LightningQubit doesn't support sampling yet
                samples = self.sample(observable, shot_range=shot_range, bin_size=bin_size)
                return np.squeeze(np.var(samples, axis=0))

            # Initialization of state
            ket = np.ravel(self._pre_rotated_state)

            M = (
                MeasurementsC64(self.state_vector)
                if self.use_csingle
                else MeasurementsC128(self.state_vector)
            )

            if observable.name == "SparseHamiltonian":
                CSR_SparseHamiltonian = observable.sparse_matrix(wire_order=self.wires).tocsr(
                    copy=False
                )
                return M.var(
                    CSR_SparseHamiltonian.indptr,
                    CSR_SparseHamiltonian.indices,
                    CSR_SparseHamiltonian.data,
                )

            if (
                observable.name in ["Hamiltonian", "Hermitian"]
                or (observable.arithmetic_depth > 0)
                or isinstance(observable.name, List)
            ):
                ob_serialized = _serialize_ob(
                    observable, self.wire_map, use_csingle=self.use_csingle
                )
                return M.var(ob_serialized)

            # translate to wire labels used by device
            observable_wires = self.map_wires(observable.wires)

            return M.var(observable.name, observable_wires)

        def generate_samples(self):
            """Generate samples

            Returns:
                array[int]: array of samples in binary representation with shape ``(dev.shots, dev.num_wires)``
            """
            M = (
                MeasurementsC64(self._kokkos_state)
                if self.use_csingle
                else MeasurementsC128(self._kokkos_state)
            )
            return M.generate_samples(len(self.wires), self.shots).astype(int, copy=False)

        def sample(self, observable, shot_range=None, bin_size=None, counts=False):
            if observable.name != "PauliZ":
                self.apply_lightning(observable.diagonalizing_gates())
                self._samples = self.generate_samples()
            return super().sample(
                observable, shot_range=shot_range, bin_size=bin_size, counts=counts
            )

        @staticmethod
        def _check_adjdiff_supported_measurements(measurements: List[MeasurementProcess]):
            """Check whether given list of measurement is supported by adjoint_differentiation.

            Args:
                measurements (List[MeasurementProcess]): a list of measurement processes to check.

            Returns:
                Expectation or State: a common return type of measurements.
            """
            if not measurements:
                return None

            if len(measurements) == 1 and measurements[0].return_type is State:
                # return State
                raise QuantumFunctionError("Not supported")

            # Now the return_type of measurement processes must be expectation
            if any(m.return_type is not Expectation for m in measurements):
                raise QuantumFunctionError(
                    "Adjoint differentiation method does not support expectation return type "
                    "mixed with other return types"
                )

            for m in measurements:
                if isinstance(m.obs, Tensor):
                    if any(isinstance(o, Projector) for o in m.obs.non_identity_obs):
                        raise QuantumFunctionError(
                            "Adjoint differentiation method does not support the Projector observable"
                        )
                elif isinstance(m.obs, Projector):
                    raise QuantumFunctionError(
                        "Adjoint differentiation method does not support the Projector observable"
                    )
            return Expectation

        @staticmethod
        def _check_adjdiff_supported_operations(operations):
            """Check Lightning adjoint differentiation method support for a tape.

            Raise ``QuantumFunctionError`` if ``tape`` contains not supported measurements,
            observables, or operations by the Lightning adjoint differentiation method.

            Args:
                tape (.QuantumTape): quantum tape to differentiate.
            """
            for op in operations:
                if op.num_params > 1 and not isinstance(op, Rot):
                    raise QuantumFunctionError(
                        f"The {op.name} operation is not supported using "
                        'the "adjoint" differentiation method'
                    )

        def _init_process_jacobian_tape(self, tape, starting_state, use_device_state):
            # Initialization of state
            if starting_state is not None:
                if starting_state.size != 2 ** len(self.wires):
                    raise QuantumFunctionError(
                        "The number of qubits of starting_state must be the same as "
                        "that of the device."
                    )
                self._apply_state_vector(starting_state, self.wires)
            elif not use_device_state:
                self.reset()
                self.apply(tape.operations)
            return self.state_vector

        def adjoint_jacobian(self, tape, starting_state=None, use_device_state=False):
            if self.shots is not None:
                warn(
                    "Requested adjoint differentiation to be computed with finite shots."
                    " The derivative is always exact when using the adjoint differentiation method.",
                    UserWarning,
                )

            tape_return_type = self._check_adjdiff_supported_measurements(tape.measurements)

            if not tape_return_type:  # the tape does not have measurements
                return np.array([], dtype=self.state.dtype)

            if tape_return_type is State:
                raise QuantumFunctionError(
                    "This method does not support statevector return type. "
                    "Use vjp method instead for this purpose."
                )

            self._check_adjdiff_supported_operations(tape.operations)

            processed_data = self._process_jacobian_tape(tape, starting_state, use_device_state)

            if not processed_data:  # training_params is empty
                return np.array([], dtype=self.state.dtype)

            trainable_params = processed_data["tp_shift"]

            # If requested batching over observables, chunk into OMP_NUM_THREADS sized chunks.
            # This will allow use of Lightning with adjoint for large-qubit numbers AND large
            # numbers of observables, enabling choice between compute time and memory use.
            requested_threads = int(getenv("OMP_NUM_THREADS", "1"))

            adjoint_jacobian = AdjointJacobianC64() if self.use_csingle else AdjointJacobianC128()

            if self._batch_obs and requested_threads > 1:
                obs_partitions = _chunk_iterable(
                    processed_data["obs_serialized"], requested_threads
                )
                jac = []
                for obs_chunk in obs_partitions:
                    jac_local = adjoint_jacobian(
                        processed_data["state_vector"],
                        obs_chunk,
                        processed_data["ops_serialized"],
                        trainable_params,
                    )
                    jac.extend(jac_local)
            else:
                jac = adjoint_jacobian(
                    processed_data["state_vector"],
                    processed_data["obs_serialized"],
                    processed_data["ops_serialized"],
                    trainable_params,
                )
            jac = np.array(jac)
            jac = jac.reshape(-1, len(trainable_params))
            jac_r = np.zeros((jac.shape[0], processed_data["all_params"]))
            jac_r[:, processed_data["record_tp_rows"]] = jac
            return self._adjoint_jacobian_processing(jac_r) if qml.active_return() else jac_r

        def vjp(self, measurements, dy, starting_state=None, use_device_state=False):
            """Generate the processing function required to compute the vector-Jacobian products of a tape.

            This function can be used with multiple expectation values or a quantum state. When a quantum state
            is given,

            .. code-block:: python

                vjp_f = dev.vjp([qml.state()], dy)
                vjp = vjp_f(tape)

            computes :math:`w = (w_1,\\cdots,w_m)` where

            .. math::

                w_k = \\langle v| \\frac{\\partial}{\\partial \\theta_k} | \\psi_{\\pmb{\\theta}} \\rangle.

            Here, :math:`m` is the total number of trainable parameters, :math:`\\pmb{\\theta}` is the vector of trainable parameters and :math:`\\psi_{\\pmb{\\theta}}`
            is the output quantum state.

            Args:
                measurements (list): List of measurement processes for vector-Jacobian product. Now it must be expectation values or a quantum state.
                dy (tensor_like): Gradient-output vector. Must have shape matching the output shape of the corresponding tape, i.e. number of measurements if the return type is expectation or :math:`2^N` if the return type is statevector
                starting_state (tensor_like): post-forward pass state to start execution with. It should be
                    complex-valued. Takes precedence over ``use_device_state``.
                use_device_state (bool): use current device state to initialize. A forward pass of the same
                    circuit should be the last thing the device has executed. If a ``starting_state`` is
                    provided, that takes precedence.

            Returns:
                The processing function required to compute the vector-Jacobian products of a tape.
            """
            if self.shots is not None:
                warn(
                    "Requested adjoint differentiation to be computed with finite shots."
                    " The derivative is always exact when using the adjoint differentiation method.",
                    UserWarning,
                )

            tape_return_type = self._check_adjdiff_supported_measurements(measurements)

            if math.allclose(dy, 0) or tape_return_type is None:
                return lambda tape: math.convert_like(np.zeros(len(tape.trainable_params)), dy)

            if tape_return_type is Expectation:
                if len(dy) != len(measurements):
                    raise ValueError(
                        "Number of observables in the tape must be the same as the length of dy in the vjp method"
                    )

                if np.iscomplexobj(dy):
                    raise ValueError(
                        "The vjp method only works with a real-valued dy when the tape is returning an expectation value"
                    )

                ham = qml.Hamiltonian(dy, [m.obs for m in measurements])

                def processing_fn(tape):
                    nonlocal ham
                    num_params = len(tape.trainable_params)

                    if num_params == 0:
                        return np.array([], dtype=self.state.dtype)

                    new_tape = tape.copy()
                    new_tape._measurements = [qml.expval(ham)]

                    return self.adjoint_jacobian(new_tape, starting_state, use_device_state)

                return processing_fn

else:

    class LightningKokkos(LightningBase):  # pragma: no cover
        name = "Lightning Kokkos PennyLane plugin [No binaries found - Fallback: default.qubit]"
        short_name = "lightning.kokkos"

        def __init__(self, wires, *, c_dtype=np.complex128, **kwargs):
            warn(
                "Pre-compiled binaries for lightning.kokkos are not available. Falling back to "
                "using the Python-based default.qubit implementation. To manually compile from "
                "source, follow the instructions at "
                "https://pennylane-lightning.readthedocs.io/en/latest/installation.html.",
                UserWarning,
            )
            super().__init__(wires, c_dtype=c_dtype, **kwargs)

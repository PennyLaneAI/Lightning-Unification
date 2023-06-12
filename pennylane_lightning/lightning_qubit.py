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
from itertools import islice, product
from warnings import warn
import numpy as np

from .lightning_base import backend_info, LightningBase, CPP_BINARY_AVAILABLE

if backend_info()["NAME"] == "lightning.qubit":
    from pennylane import (
        math,
        BasisState,
        QubitStateVector,
        Projector,
        Rot,
        DeviceError,
    )
    from pennylane.operation import Tensor, Operation
    from pennylane.measurements import Expectation, State
    from pennylane.wires import Wires

    # tolerance for numerical errors
    tolerance = 1e-10

    import pennylane as qml

    from ._version import __version__

    from .pennylane_lightning_ops import (
        # adjoint_diff,
        # MeasuresC64,
        StateVectorC64,
        # MeasuresC128,
        StateVectorC128,
    )

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
        "Projector",
        "SparseHamiltonian",
        "Hamiltonian",
        "Sum",
        "SProd",
        "Prod",
        "Exp",
    }

    class LightningQubit(LightningBase):
        """PennyLane Lightning Qubit device.

        A device that interfaces with C++ to perform fast linear algebra calculations.

        Use of this device requires pre-built binaries or compilation from source. Check out the
        :doc:`/installation` guide for more details.

        Args:
            wires (int): the number of wires to initialize the device with
            c_dtype: Datatypes for statevector representation. Must be one of ``np.complex64`` or ``np.complex128``.
            shots (int): How many times the circuit should be evaluated (or sampled) to estimate
                the expectation values. Defaults to ``None`` if not specified. Setting
                to ``None`` results in computing statistics like expectation values and
                variances analytically.
            mcmc (bool): Determine whether to use the approximate Markov Chain Monte Carlo sampling method when generating samples.
            kernel_name (str): name of transition kernel. The current version supports two kernels: ``"Local"`` and ``"NonZeroRandom"``.
                The local kernel conducts a bit-flip local transition between states. The local kernel generates a
                random qubit site and then generates a random number to determine the new bit at that qubit site. The ``"NonZeroRandom"`` kernel
                randomly transits between states that have nonzero probability.
            num_burnin (int): number of steps that will be dropped. Increasing this value will
                result in a closer approximation but increased runtime.
            batch_obs (bool): Determine whether we process observables in parallel when computing the
                jacobian. This value is only relevant when the lightning qubit is built with OpenMP.
        """

        name = "Lightning Qubit PennyLane plugin"
        short_name = "lightning.qubit"
        operations = allowed_operations
        observables = allowed_observables

        def __init__(
            self,
            wires,
            *,
            c_dtype=np.complex128,
            shots=None,
            mcmc=False,
            kernel_name="Local",
            num_burnin=100,
            batch_obs=False,
            analytic=None,
        ):
            super().__init__(wires, shots=shots, c_dtype=c_dtype, analytic=analytic)

            # Create the initial state. Internally, we store the
            # state as an array of dimension [2]*wires.
            self._state = self._create_basis_state(0)
            self._pre_rotated_state = self._state

            self._mcmc = mcmc
            if self._mcmc:
                if kernel_name not in [
                    "Local",
                    "NonZeroRandom",
                ]:
                    raise NotImplementedError(
                        f"The {kernel_name} is not supported and currently only 'Local' and 'NonZeroRandom' kernels are supported."
                    )
                if num_burnin >= shots:
                    raise ValueError("Shots should be greater than num_burnin.")
                self._kernel_name = kernel_name
                self._num_burnin = num_burnin

        def _create_basis_state(self, index):
            """Return a computational basis state over all wires.
            Args:
                index (int): integer representing the computational basis state
            Returns:
                array[complex]: complex array of shape ``[2]*self.num_wires``
                representing the statevector of the basis state
            Note: This function does not support broadcasted inputs yet.
            """
            state = np.zeros(2**self.num_wires, dtype=np.complex128)
            state[index] = 1
            state = self._asarray(state, dtype=self.C_DTYPE)
            return self._reshape(state, [2] * self.num_wires)

        def reset(self):
            """Reset the device"""
            super().reset()

            # init the state vector to |00..0>
            self._state = self._create_basis_state(0)
            self._pre_rotated_state = self._state

        @property
        def state(self):
            # Flattening the state.
            shape = (1 << self.num_wires,)
            return self._reshape(self._pre_rotated_state, shape)

        def _apply_state_vector(self, state, device_wires):
            """Initialize the internal state vector in a specified state.
            Args:
                state (array[complex]): normalized input state of length ``2**len(wires)``
                    or broadcasted state of shape ``(batch_size, 2**len(wires))``
                device_wires (Wires): wires that get initialized in the state
            """

            # translate to wire labels used by device
            device_wires = self.map_wires(device_wires)

            state = self._asarray(state, dtype=self.C_DTYPE)
            output_shape = [2] * self.num_wires

            if len(device_wires) == self.num_wires and Wires(sorted(device_wires)) == device_wires:
                # Initialize the entire device state with the input state
                self._state = self._reshape(state, output_shape)
                return

            # generate basis states on subset of qubits via the cartesian product
            basis_states = np.array(list(product([0, 1], repeat=len(device_wires))))

            # get basis states to alter on full set of qubits
            unravelled_indices = np.zeros((2 ** len(device_wires), self.num_wires), dtype=int)
            unravelled_indices[:, device_wires] = basis_states

            # get indices for which the state is changed to input state vector elements
            ravelled_indices = np.ravel_multi_index(unravelled_indices.T, [2] * self.num_wires)

            state = self._scatter(ravelled_indices, state, [2**self.num_wires])
            state = self._reshape(state, output_shape)
            self._state = self._asarray(state, dtype=self.C_DTYPE)

        def _apply_basis_state(self, state, wires):
            """Initialize the state vector in a specified computational basis state.

            Args:
                state (array[int]): computational basis state of shape ``(wires,)``
                    consisting of 0s and 1s.
                wires (Wires): wires that the provided computational state should be initialized on

            Note: This function does not support broadcasted inputs yet.
            """
            # translate to wire labels used by device
            device_wires = self.map_wires(wires)

            # length of basis state parameter
            n_basis_state = len(state)

            if not set(state.tolist()).issubset({0, 1}):
                raise ValueError("BasisState parameter must consist of 0 or 1 integers.")

            if n_basis_state != len(device_wires):
                raise ValueError("BasisState parameter and wires must be of equal length.")

            # get computational basis state number
            basis_states = 2 ** (self.num_wires - 1 - np.array(device_wires))
            basis_states = qml.math.convert_like(basis_states, state)
            num = int(qml.math.dot(state, basis_states))

            self._state = self._create_basis_state(num)

        def apply_lightning(self, state, operations):
            """Apply a list of operations to the state tensor.

            Args:
                state (array[complex]): the input state tensor
                operations (list[~pennylane.operation.Operation]): operations to apply
                dtype (type): Type of numpy ``complex`` to be used. Can be important
                to specify for large systems for memory allocation purposes.

            Returns:
                array[complex]: the output state tensor
            """
            state_vector = np.ravel(state)

            if self.use_csingle:
                # use_csingle
                sim = StateVectorC64(state_vector)
            else:
                # self.C_DTYPE is np.complex128 by default
                sim = StateVectorC128(state_vector)

            # Skip over identity operations instead of performing
            # matrix multiplication with the identity.
            skipped_ops = ["Identity"]

            for o in operations:
                if o.name in skipped_ops:
                    continue
                method = getattr(sim, o.name, None)

                wires = self.wires.indices(o.wires)
                if method is None:
                    # Inverse can be set to False since qml.matrix(o) is already in inverted form
                    method = getattr(sim, "applyMatrix")
                    try:
                        method(qml.matrix(o), wires, False)
                    except AttributeError:  # pragma: no cover
                        # To support older versions of PL
                        method(o.matrix, wires, False)
                else:
                    inv = False
                    param = o.parameters
                    method(wires, inv, param)

            return np.reshape(state_vector, state.shape)

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
                        "Operation {} cannot be used after other Operations have already been "
                        "applied on a {} device.".format(operation.name, self.short_name)
                    )

            if operations:
                self._pre_rotated_state = self.apply_lightning(self._state, operations)
            else:
                self._pre_rotated_state = self._state

            if rotations:
                self._state = self.apply_lightning(np.copy(self._pre_rotated_state), rotations)
            else:
                self._state = self._pre_rotated_state

else:

    class LightningQubit(LightningBase):  # pragma: no cover
        name = "Lightning Qubit PennyLane plugin [No binaries found - Fallback: default.qubit]"
        short_name = "lightning.qubit"

        def __init__(self, wires, *, c_dtype=np.complex128, **kwargs):
            warn(
                "Pre-compiled binaries for lightning.qubit are not available. Falling back to "
                "using the Python-based default.qubit implementation. To manually compile from "
                "source, follow the instructions at "
                "https://pennylane-lightning.readthedocs.io/en/latest/installation.html.",
                UserWarning,
            )
            super().__init__(wires, c_dtype=c_dtype, **kwargs)

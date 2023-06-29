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
This module contains the base class for all PennyLane Lightning simulator devices,
and interfaces with C++ for improved performance.
"""
import numpy as np

from ._version import __version__

try:
    from .pennylane_lightning_ops import (
        MeasurementsC64,
        StateVectorC64,
        MeasurementsC128,
        StateVectorC128,
        backend_info,
    )

    CPP_BINARY_AVAILABLE = True
except ModuleNotFoundError:

    def backend_info():
        return {"NAME": "NONE"}

    CPP_BINARY_AVAILABLE = False

if CPP_BINARY_AVAILABLE:
    from typing import List

    import pennylane as qml
    from pennylane import QubitDevice
    from pennylane.operation import Operation
    from pennylane.wires import Wires

    class LightningBase(QubitDevice):
        """PennyLane Lightning Base device.

        This intermediate base class provides device-agnostic functionalities.

        Use of this device requires pre-built binaries or compilation from source. Check out the
        :doc:`/installation` guide for more details.

        Args:
            wires (int): the number of wires to initialize the device with
            c_dtype: Datatypes for statevector representation. Must be one of ``np.complex64`` or ``np.complex128``.
            shots (int): How many times the circuit should be evaluated (or sampled) to estimate
                stochastic return values. Defaults to ``None`` if not specified. Setting
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
        ):
            if c_dtype is np.complex64:
                r_dtype = np.float32
                self.use_csingle = True
            elif c_dtype is np.complex128:
                r_dtype = np.float64
                self.use_csingle = False
            else:
                raise TypeError(f"Unsupported complex Type: {c_dtype}")
            super().__init__(wires, shots=shots, r_dtype=r_dtype, c_dtype=c_dtype)
            self._batch_obs = batch_obs

        @property
        def stopping_condition(self):
            """.BooleanFn: Returns the stopping condition for the device. The returned
            function accepts a queueable object (including a PennyLane operation
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

        def probability(self, wires=None, shot_range=None, bin_size=None):
            """Return the probability of each computational basis state.

            Devices that require a finite number of shots always return the
            estimated probability.

            Args:
                wires (Iterable[Number, str], Number, str, Wires): wires to return
                    marginal probabilities for. Wires not provided are traced out of the system.
                shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
                    to use. If not specified, all samples are used.
                bin_size (int): Divides the shot range into bins of size ``bin_size``, and
                    returns the measurement statistic separately over each bin. If not
                    provided, the entire shot range is treated as a single bin.

            Returns:
                array[float]: list of the probabilities
            """
            if self.shots is not None:
                return self.estimate_probability(
                    wires=wires, shot_range=shot_range, bin_size=bin_size
                )

            wires = wires or self.wires
            wires = Wires(wires)

            # translate to wire labels used by device
            device_wires = self.map_wires(wires)

            if (
                device_wires
                and len(device_wires) > 1
                and (not np.all(np.array(device_wires)[:-1] <= np.array(device_wires)[1:]))
            ):
                raise RuntimeError(
                    "Lightning does not currently support out-of-order indices for probabilities"
                )

            ket = np.ravel(self._state)

            state_vector = StateVectorC64(ket) if self.use_csingle else StateVectorC128(ket)
            M = (
                MeasurementsC64(state_vector)
                if self.use_csingle
                else MeasurementsC128(state_vector)
            )
            return M.probs(device_wires)

        def _get_diagonalizing_gates(self, circuit: qml.tape.QuantumTape) -> List[Operation]:
            skip_diagonalizing = lambda obs: isinstance(obs, qml.Hamiltonian) or (
                isinstance(obs, qml.ops.Sum) and obs._pauli_rep is not None
            )
            meas_filtered = list(
                filter(
                    lambda m: m.obs is None or not skip_diagonalizing(m.obs), circuit.measurements
                )
            )
            return super()._get_diagonalizing_gates(
                qml.tape.QuantumScript(measurements=meas_filtered)
            )

else:  # binaries not available
    from pennylane.devices import DefaultQubit

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

# Copyright 2021 Xanadu Quantum Technologies Inc.

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
Helper functions for serializing quantum tapes.
"""
from typing import List, Tuple
import numpy as np
from pennylane import (
    BasisState,
    Hadamard,
    PauliX,
    PauliY,
    PauliZ,
    Identity,
    QubitStateVector,
    Rot,
)
from pennylane.operation import Tensor
from pennylane.tape import QuantumTape
from pennylane.math import unwrap

from pennylane import matrix, DeviceError

pauli_name_map = {
    "I": "Identity",
    "X": "PauliX",
    "Y": "PauliY",
    "Z": "PauliZ",
}


class _Serialize:
    def __init__(self, device_name):
        if device_name == "lightning.qubit":
            try:
                from .lightning_qubit_ops import (
                    StateVectorC128,
                )
                from .lightning_qubit_ops.observables import (
                    NamedObsC64,
                    NamedObsC128,
                    HermitianObsC64,
                    HermitianObsC128,
                    TensorProdObsC64,
                    TensorProdObsC128,
                    HamiltonianC64,
                    HamiltonianC128,
                )
            except ImportError:
                raise ImportError(
                    "Pre-compiled binaries for "
                    + device_name
                    + " serialize functionality are not available."
                )
        elif device_name == "lightning.kokkos":
            try:
                from .lightning_kokkos_ops import (
                    StateVectorC128,
                )
                from .lightning_kokkos_ops.observables import (
                    NamedObsC64,
                    NamedObsC128,
                    HermitianObsC64,
                    HermitianObsC128,
                    TensorProdObsC64,
                    TensorProdObsC128,
                    HamiltonianC64,
                    HamiltonianC128,
                )
            except ImportError:
                raise ImportError(
                    "Pre-compiled binaries for "
                    + device_name
                    + " serialize functionality are not available."
                )
        else:
            raise DeviceError('The device name "' + device_name + '" is not a valid option.')
        self.StateVectorC128 = StateVectorC128
        self.NamedObsC64 = NamedObsC64
        self.NamedObsC128 = NamedObsC128
        self.HermitianObsC64 = HermitianObsC64
        self.HermitianObsC128 = HermitianObsC128
        self.TensorProdObsC64 = TensorProdObsC64
        self.TensorProdObsC128 = TensorProdObsC128
        self.HamiltonianC64 = HamiltonianC64
        self.HamiltonianC128 = HamiltonianC128

    def _named_obs(self, ob, wires_map: dict, use_csingle: bool):
        """Serializes a Named observable"""
        named_obs = self.NamedObsC64 if use_csingle else self.NamedObsC128
        wires = [wires_map[w] for w in ob.wires]
        if ob.name == "Identity":
            wires = wires[:1]
        return named_obs(ob.name, wires)

    def _hermitian_ob(self, o, wires_map: dict, use_csingle: bool):
        """Serializes a Hermitian observable"""
        assert not isinstance(o, Tensor)

        if use_csingle:
            ctype = np.complex64
            hermitian_obs = self.HermitianObsC64
        else:
            ctype = np.complex128
            hermitian_obs = self.HermitianObsC128

        wires = [wires_map[w] for w in o.wires]
        return hermitian_obs(matrix(o).ravel().astype(ctype), wires)

    def _tensor_ob(self, ob, wires_map: dict, use_csingle: bool):
        """Serialize a tensor observable"""
        assert isinstance(ob, Tensor)

        if use_csingle:
            tensor_obs = self.TensorProdObsC64
        else:
            tensor_obs = self.TensorProdObsC128

        return tensor_obs([self._ob(o, wires_map, use_csingle) for o in ob.obs])

    def _hamiltonian(self, ob, wires_map: dict, use_csingle: bool):
        if use_csingle:
            rtype = np.float32
            hamiltonian_obs = self.HamiltonianC64
        else:
            rtype = np.float64
            hamiltonian_obs = self.HamiltonianC128

        coeffs = np.array(unwrap(ob.coeffs)).astype(rtype)
        terms = [self._ob(t, wires_map, use_csingle) for t in ob.ops]
        return hamiltonian_obs(coeffs, terms)

    def _pauli_word(self, ob, wires_map: dict, use_csingle: bool):
        """Serialize a :class:`pennylane.pauli.PauliWord` into a Named or Tensor observable."""
        if use_csingle:
            named_obs = self.NamedObsC64
            tensor_obs = self.TensorProdObsC64
        else:
            named_obs = self.NamedObsC128
            tensor_obs = self.TensorProdObsC128

        if len(ob) == 1:
            wire, pauli = list(ob.items())[0]
            return named_obs(pauli_name_map[pauli], [wires_map[wire]])

        return tensor_obs(
            [named_obs(pauli_name_map[pauli], [wires_map[wire]]) for wire, pauli in ob.items()]
        )

    def _pauli_sentence(self, ob, wires_map: dict, use_csingle: bool):
        """Serialize a :class:`pennylane.pauli.PauliSentence` into a Hamiltonian."""
        if use_csingle:
            rtype = np.float32
            hamiltonian_obs = self.HamiltonianC64
        else:
            rtype = np.float64
            hamiltonian_obs = self.HamiltonianC128

        pwords, coeffs = zip(*ob.items())
        terms = [self._pauli_word(pw, wires_map, use_csingle) for pw in pwords]
        coeffs = np.array(coeffs).astype(rtype)
        return hamiltonian_obs(coeffs, terms)

    def _ob(self, ob, wires_map, use_csingle):
        """Serialize a :class:`pennylane.operation.Observable` into an Observable."""
        if isinstance(ob, Tensor):
            return self._tensor_ob(ob, wires_map, use_csingle)
        if ob.name == "Hamiltonian":
            return self._hamiltonian(ob, wires_map, use_csingle)
        if isinstance(ob, (PauliX, PauliY, PauliZ, Identity, Hadamard)):
            return self._named_obs(ob, wires_map, use_csingle)
        if ob._pauli_rep is not None:
            return self._pauli_sentence(ob._pauli_rep, wires_map, use_csingle)
        return self._hermitian_ob(ob, wires_map, use_csingle)

    def _observables(self, tape: QuantumTape, wires_map: dict, use_csingle: bool = False) -> List:
        """Serializes the observables of an input tape.

        Args:
            tape (QuantumTape): the input quantum tape
            wires_map (dict): a dictionary mapping input wires to the device's backend wires
            use_csingle (bool): whether to use np.complex64 instead of np.complex128

        Returns:
            list(ObsStructC128 or ObsStructC64): A list of observable objects compatible with the C++ backend
        """

        return [self._ob(ob, wires_map, use_csingle) for ob in tape.observables]

    def _ops(
        self, tape: QuantumTape, wires_map: dict
    ) -> Tuple[List[List[str]], List[np.ndarray], List[List[int]], List[bool], List[np.ndarray]]:
        """Serializes the operations of an input tape.

        The state preparation operations are not included.

        Args:
            tape (QuantumTape): the input quantum tape
            wires_map (dict): a dictionary mapping input wires to the device's backend wires

        Returns:
            Tuple[list, list, list, list, list]: A serialization of the operations, containing a list
            of operation names, a list of operation parameters, a list of observable wires, a list of
            inverses, and a list of matrices for the operations that do not have a dedicated kernel.
        """
        names = []
        params = []
        wires = []
        mats = []

        uses_stateprep = False

        for o in tape.operations:
            if isinstance(o, (BasisState, QubitStateVector)):
                uses_stateprep = True
                continue
            elif isinstance(o, Rot):
                op_list = o.expand().operations
            else:
                op_list = [o]

            for single_op in op_list:
                name = single_op.name
                names.append(name)

                if not hasattr(self.StateVectorC128, name):
                    params.append([])
                    mats.append(matrix(single_op))

                else:
                    params.append(single_op.parameters)
                    mats.append([])

                wires_list = single_op.wires.tolist()
                wires.append([wires_map[w] for w in wires_list])

        inverses = [False] * len(names)
        return (names, params, wires, inverses, mats), uses_stateprep

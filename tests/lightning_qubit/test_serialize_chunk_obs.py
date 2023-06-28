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
"""
Unit tests for the serialization helper functions.
"""
import pennylane as qml
import numpy as np
import pennylane_lightning

from pennylane_lightning._serialize import (
    _serialize_observables,
    _serialize_ops,
    _serialize_ob,
)
import pytest
from unittest import mock

from pennylane_lightning import CPP_BINARY_AVAILABLE, backend_info

if not CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

if backend_info()["NAME"] != "lightning.qubit":
    pytest.skip("Exclusive tests for lightning.qubit. Skipping.", allow_module_level=True)


class TestSerializeObs:
    """Tests for the _serialize_observables function"""

    wires_dict = {i: i for i in range(10)}

    @pytest.mark.parametrize("use_csingle", [True, False])
    @pytest.mark.parametrize("ObsChunk", list(range(1, 5)))
    def test_chunk_obs(self, use_csingle, ObsChunk):
        """Test chunking of observable array"""
        with qml.tape.QuantumTape() as tape:
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))
            qml.expval(qml.PauliY(wires=1))
            qml.expval(qml.PauliX(0) @ qml.Hermitian([[0, 1], [1, 0]], wires=3) @ qml.Hadamard(2))
            qml.expval(qml.Hermitian(qml.PauliZ.compute_matrix(), wires=0) @ qml.Identity(1))

        s = _serialize_observables(tape, self.wires_dict, use_csingle=use_csingle)

        obtained_chunks = pennylane_lightning.lightning_qubit._chunk_iterable(s, ObsChunk)
        assert len(list(obtained_chunks)) == int(np.ceil(len(s) / ObsChunk))

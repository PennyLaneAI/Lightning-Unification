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
Unit tests for Sparse Measurements in lightning.qubit.
"""
import numpy as np
import pennylane as qml
from pennylane import qchem

import pytest

from pennylane_lightning import CPP_BINARY_AVAILABLE, backend_info

if not CPP_BINARY_AVAILABLE:
    pytest.skip("No binary module found. Skipping.", allow_module_level=True)

if backend_info()["NAME"] != "lightning.qubit":
    pytest.skip("Exclusive tests for lightning.qubit. Skipping.", allow_module_level=True)


class TestSparseExpval:
    """Tests for the expval function"""

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return qml.device("lightning.qubit", wires=2, c_dtype=request.param)

    @pytest.mark.skipif(
        backend_info()["USE_KOKKOS"], reason="Kokkos and Kokkos Kernels are present."
    )
    def test_create_device_with_unsupported_dtype(self, dev):
        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.RY(-0.2, wires=[1])
            return qml.expval(
                qml.SparseHamiltonian(
                    qml.Hamiltonian([1], [qml.PauliX(0) @ qml.Identity(1)]).sparse_matrix(),
                    wires=[0, 1],
                )
            )

        with pytest.raises(
            NotImplementedError,
            match="The expval of a SparseHamiltonian requires Kokkos and Kokkos Kernels.",
        ):
            circuit()

    @pytest.mark.parametrize(
        "cases",
        [
            [qml.PauliX(0) @ qml.Identity(1), 0.00000000000000000],
            [qml.Identity(0) @ qml.PauliX(1), -0.19866933079506122],
            [qml.PauliY(0) @ qml.Identity(1), -0.38941834230865050],
            [qml.Identity(0) @ qml.PauliY(1), 0.00000000000000000],
            [qml.PauliZ(0) @ qml.Identity(1), 0.92106099400288520],
            [qml.Identity(0) @ qml.PauliZ(1), 0.98006657784124170],
        ],
    )
    @pytest.mark.skipif(
        not backend_info()["USE_KOKKOS"], reason="Requires Kokkos and Kokkos Kernels."
    )
    def test_sparse_Pauli_words(self, cases, tol, dev):
        """Test expval of some simple sparse Hamiltonian"""

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.RX(0.4, wires=[0])
            qml.RY(-0.2, wires=[1])
            return qml.expval(
                qml.SparseHamiltonian(
                    qml.Hamiltonian([1], [cases[0]]).sparse_matrix(), wires=[0, 1]
                )
            )

        assert np.allclose(circuit(), cases[1], atol=tol, rtol=0)


class TestSparseExpvalQChem:
    """Tests for the expval function with qchem workflow"""

    symbols = ["Li", "H"]
    geometry = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.969280527])

    H, qubits = qchem.molecular_hamiltonian(
        symbols,
        geometry,
    )
    H_sparse = H.sparse_matrix()

    active_electrons = 1

    hf_state = qchem.hf_state(active_electrons, qubits)

    singles, doubles = qchem.excitations(active_electrons, qubits)
    excitations = singles + doubles

    @pytest.fixture(params=[np.complex64, np.complex128])
    def dev(self, request):
        return qml.device("lightning.qubit", wires=12, c_dtype=request.param)

    @pytest.mark.parametrize(
        "qubits, wires, H_sparse, hf_state, excitations",
        [
            [qubits, range(qubits), H_sparse, hf_state, excitations],
            [
                qubits,
                np.random.permutation(np.arange(qubits)),
                H_sparse,
                hf_state,
                excitations,
            ],
        ],
    )
    @pytest.mark.skipif(
        not backend_info()["USE_KOKKOS"], reason="Requires Kokkos and Kokkos Kernels."
    )
    def test_sparse_Pauli_words(self, qubits, wires, H_sparse, hf_state, excitations, tol, dev):
        """Test expval of some simple sparse Hamiltonian"""

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.BasisState(hf_state, wires=range(qubits))

            for excitation in excitations:
                if len(excitation) == 4:
                    qml.DoubleExcitation(1, wires=excitation)
                elif len(excitation) == 2:
                    qml.SingleExcitation(1, wires=excitation)

            return qml.expval(qml.SparseHamiltonian(H_sparse, wires=wires))

        dev_default = qml.device("default.qubit", wires=qubits)

        @qml.qnode(dev_default, diff_method="parameter-shift")
        def circuit_default():
            qml.BasisState(hf_state, wires=range(qubits))

            for excitation in excitations:
                if len(excitation) == 4:
                    qml.DoubleExcitation(1, wires=excitation)
                elif len(excitation) == 2:
                    qml.SingleExcitation(1, wires=excitation)

            return qml.expval(qml.SparseHamiltonian(H_sparse, wires=wires))

        assert np.allclose(circuit(), circuit_default(), atol=tol, rtol=0)

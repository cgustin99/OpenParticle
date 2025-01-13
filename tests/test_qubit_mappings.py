from openparticle import ParticleOperator, Fock
from openparticle.utils import get_fock_basis, generate_matrix, _check_cutoff
from symmer import PauliwordOp, QuantumState
import pytest
import numpy as np


def _verify_mapping(
    op,
    state,
    max_fermionic_mode: int = None,
    max_antifermionic_mode: int = None,
    max_bosonic_mode: int = None,
    max_bosonic_occupancy: int = None,
    threshold: float = 1e-14,
):
    n_qubits = sum(
        state.allocate_qubits(
            max_fermionic_mode=max_fermionic_mode,
            max_antifermionic_mode=max_antifermionic_mode,
            max_bosonic_mode=max_bosonic_mode,
            max_bosonic_occupancy=max_bosonic_occupancy,
        )
    )
    output = _check_cutoff(op * state, max_bosonic_occupancy=max_bosonic_occupancy)

    output_pauli = op.to_paulis(
        max_fermionic_mode=max_fermionic_mode,
        max_antifermionic_mode=max_antifermionic_mode,
        max_bosonic_mode=max_bosonic_mode,
        max_bosonic_occupancy=max_bosonic_occupancy,
    ) * state.to_qubit_state(
        max_fermionic_mode=max_fermionic_mode,
        max_antifermionic_mode=max_antifermionic_mode,
        max_bosonic_mode=max_bosonic_mode,
        max_bosonic_occupancy=max_bosonic_occupancy,
    )
    if isinstance(output, Fock):
        expected_output = output.to_qubit_state(
            max_fermionic_mode=max_fermionic_mode,
            max_antifermionic_mode=max_antifermionic_mode,
            max_bosonic_mode=max_bosonic_mode,
            max_bosonic_occupancy=max_bosonic_occupancy,
        )
    else:
        expected_output = QuantumState([0] * n_qubits) * 0

    similarity = output_pauli - expected_output
    # print("-----")
    # print("Initial state: ", state)
    # print("Final state (output): ", output)
    # print("output pauli: ", output_pauli)
    # print("expected: ", expected_output)
    # print("T or F: ", (list(similarity.state_op.coeff_vec) == []))
    if np.sum(similarity.state_op.coeff_vec) > threshold:
        assert list(similarity.state_op.coeff_vec) == []


def test_JW_dagger():
    op = ParticleOperator("b0^")
    jw_op = PauliwordOp.from_dictionary({"X": 1 / 2, "Y": -1j / 2})
    assert op.to_paulis(max_fermionic_mode=0) == jw_op


def test_JW():
    op = ParticleOperator("b0")
    jw_op = PauliwordOp.from_dictionary({"X": 1 / 2, "Y": 1j / 2})
    assert op.to_paulis(max_fermionic_mode=0) == jw_op


def test_product_JW():
    op = ParticleOperator("b0^ b0")
    jw_op = PauliwordOp.from_dictionary({"I": 1 / 2, "Z": -1 / 2})
    assert op.to_paulis(max_fermionic_mode=0) == jw_op


def test_product_different_modes_JW():
    op = ParticleOperator("b0^ b1")
    jw_op = PauliwordOp.from_dictionary(
        {"XX": 1 / 4, "XY": -1j / 4, "YX": 1j / 4, "YY": 1 / 4}
    )
    assert op.to_paulis(max_fermionic_mode=1) == jw_op


def test_spread_JW():
    op = ParticleOperator("b3^ b0")
    jw_op = PauliwordOp.from_dictionary(
        {"XZZY": 0.25j, "YZZY": (0.25 + 0j), "XZZX": (0.25 + 0j), "YZZX": -0.25j}
    )
    assert op.to_paulis(max_fermionic_mode=3) == jw_op


def test_JW_sum():
    op = ParticleOperator("b0") + ParticleOperator("b1")
    assert op.to_paulis(max_fermionic_mode=1) == PauliwordOp.from_dictionary(
        {"IX": 0.5, "IY": 0.5j, "XZ": 0.5, "YZ": 0.5j}
    )


def test_SB():
    op = ParticleOperator("a0")
    SB_op = PauliwordOp.from_dictionary({"X": 1 / 2, "Y": 1j / 2})
    assert op.to_paulis(max_bosonic_mode=0, max_bosonic_occupancy=1) == SB_op


def test_SB_higher_max_occ():
    op = ParticleOperator("a0")
    SB_op = PauliwordOp.from_dictionary(
        {
            "IX": (0.6830127018922193 + 0j),
            "ZX": (-0.1830127018922193 + 0j),
            "IY": 0.6830127018922193j,
            "ZY": -0.1830127018922193j,
            "XX": (0.3535533905932738 + 0j),
            "YX": 0.3535533905932738j,
            "XY": -0.3535533905932738j,
            "YY": (0.3535533905932738 + 0j),
        }
    )
    assert op.to_paulis(max_bosonic_mode=0, max_bosonic_occupancy=3) == SB_op


@pytest.mark.parametrize(
    "op",
    [
        ParticleOperator("b0"),
        ParticleOperator("a0 a0"),
        ParticleOperator("a0 a1"),
        ParticleOperator("a0") + ParticleOperator("a2"),
        ParticleOperator("a0 a0") + ParticleOperator("a0^ a0^"),
        ParticleOperator("b0 b1"),
        ParticleOperator("b0 d0"),
        ParticleOperator("b1 d1"),
        ParticleOperator("b0 b1 d0"),
        ParticleOperator("b0 b2 b3 d2"),
    ],
)
def test_mapping(op):
    max_fermionic_mode = op.max_fermionic_mode
    max_antifermionic_mode = op.max_antifermionic_mode
    max_bosonic_mode = op.max_bosonic_mode
    max_bosonic_occupancy = 3

    basis = get_fock_basis(op, max_bose_occ=max_bosonic_occupancy)
    for state in range(len(basis)):
        _verify_mapping(
            op,
            basis[state],
            max_fermionic_mode=max_fermionic_mode,
            max_antifermionic_mode=max_antifermionic_mode,
            max_bosonic_mode=max_bosonic_mode,
            max_bosonic_occupancy=max_bosonic_occupancy,
        )


def test_mapping_product_of_different_types_ops():
    op = ParticleOperator("b1^ a0")
    max_bose_occ = 1
    qubit_op = PauliwordOp.from_dictionary(
        {"XZX": (0.25 + 0j), "YZX": -0.25j, "XZY": 0.25j, "YZY": (0.25 + 0j)}
    )
    assert (
        op.to_paulis(
            max_fermionic_mode=1, max_bosonic_mode=0, max_bosonic_occupancy=max_bose_occ
        )
        == qubit_op
    )


def test_fock_qubit_map():
    f_occ = [4, 2, 1]
    af_occ = [0]
    b_occ = [(0, 3), (2, 1)]

    max_f_mode = 5
    max_af_mode = 0
    max_b_mode = 2
    max_bose_occ = 3

    mapped_bitstring = "010110" + "1" + "01" + "00" + "11"
    expected_quantum_state = QuantumState.from_dictionary({mapped_bitstring: 1.0})
    state = Fock(f_occ, af_occ, b_occ)
    obtained_quantum_state = state.to_qubit_state(
        max_bosonic_occupancy=max_bose_occ,
        max_fermionic_mode=max_f_mode,
        max_antifermionic_mode=max_af_mode,
        max_bosonic_mode=max_b_mode,
    )

    assert expected_quantum_state == obtained_quantum_state

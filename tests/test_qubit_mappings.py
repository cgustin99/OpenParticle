from openparticle.qubit_mappings import *
from openparticle import ParticleOperator, Fock
from symmer import PauliwordOp, QuantumState


def test_JW_dagger():
    op = ParticleOperator("b0^")
    jw_op = PauliwordOp.from_dictionary({"X": 1 / 2, "Y": -1j / 2})
    assert jordan_wigner(op) == jw_op


def test_JW():
    op = ParticleOperator("b0")
    jw_op = PauliwordOp.from_dictionary({"X": 1 / 2, "Y": 1j / 2})
    assert jordan_wigner(op) == jw_op


def test_product_JW():
    op = ParticleOperator("b0^ b0")
    jw_op = PauliwordOp.from_dictionary({"I": 1 / 2, "Z": -1 / 2})
    assert jordan_wigner(op) == jw_op


def test_product_different_modes_JW():
    op = ParticleOperator("b0^ b1")
    jw_op = PauliwordOp.from_dictionary(
        {"XX": 1 / 4, "XY": -1j / 4, "YX": 1j / 4, "YY": 1 / 4}
    )
    assert jordan_wigner(op) == jw_op


def test_spread_JW():
    op = ParticleOperator("b3^ b0")
    jw_op = PauliwordOp.from_dictionary(
        {"XZZY": 0.25j, "YZZY": (0.25 + 0j), "XZZX": (0.25 + 0j), "YZZX": -0.25j}
    )
    assert jordan_wigner(op) == jw_op


def test_SB():
    op = ParticleOperator("a0")
    SB_op = PauliwordOp.from_dictionary({"X": 1 / 2, "Y": 1j / 2})
    assert SB(op, 1) == SB_op


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
    assert SB(op, 3) == SB_op


def test_mapping_product_of_different_types_ops():
    op = ParticleOperator("b1^ a0")
    max_bose_occ = 1
    qubit_op = PauliwordOp.from_dictionary(
        {"XZX": (0.25 + 0j), "YZX": -0.25j, "XZY": 0.25j, "YZY": (0.25 + 0j)}
    )
    assert op_qubit_map(op, max_bose_occ=max_bose_occ) == qubit_op


def test_fock_qubit_map():
    f_occ = [4, 2, 1]
    af_occ = [0]
    b_occ = [(0, 3), (2, 1)]

    max_f_mode = 5
    max_af_mode = 0
    max_b_mode = 2
    max_occ = 3

    mapped_bitstring = "010110" + "1" + "01" + "00" + "11"
    expected_quantum_state = QuantumState.from_dictionary({mapped_bitstring: 1.0})
    state = Fock(f_occ, af_occ, b_occ)
    obtained_quantum_state = fock_qubit_state_mapping(
        state=state,
        max_occ=max_occ,
        max_fermion_mode=max_f_mode,
        max_antifermion_mode=max_af_mode,
        max_boson_mode=max_b_mode,
    )

    assert expected_quantum_state == obtained_quantum_state

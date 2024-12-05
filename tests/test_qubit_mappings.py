from openparticle.qubit_mappings import *
from symmer import PauliwordOp


def test_JW_dagger():
    op = ParticleOperator("b0^")
    jw_op = PauliwordOp.from_dictionary({"X": 1 / 2, "Y": -1j / 2})
    assert op_qubit_map(op) == jw_op


def test_JW():
    op = ParticleOperator("b0")
    jw_op = PauliwordOp.from_dictionary({"X": 1 / 2, "Y": 1j / 2})
    assert op_qubit_map(op) == jw_op


def test_product_JW():
    op = ParticleOperator("b0^ b0")
    jw_op = PauliwordOp.from_dictionary({"I": 1 / 2, "Z": -1 / 2})
    assert op_qubit_map(op) == jw_op


def test_product_different_modes_JW():
    op = ParticleOperator("b0^ b1")
    jw_op = PauliwordOp.from_dictionary(
        {"XX": 1 / 4, "XY": -1j / 4, "YX": 1j / 4, "YY": 1 / 4}
    )
    assert op_qubit_map(op) == jw_op


def test_SB():
    op = ParticleOperator("a0")
    SB_op = PauliwordOp.from_dictionary({"X": 1 / 2, "Y": 1j / 2})
    assert op_qubit_map(op, 1) == SB_op


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
    assert op_qubit_map(op, 3) == SB_op

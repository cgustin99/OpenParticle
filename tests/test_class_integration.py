from openparticle import (
    ParticleOperator,
    Fock,
)
import numpy as np
from openparticle.utils import get_matrix_element
import pytest


def op_on_state_1():
    out = ParticleOperator("b1^") * Fock([2], [], [])
    assert out.op_dict == {((0, 1, 1), (0, 2, 1)): 1.0}


def op_on_state_2():
    out = ParticleOperator("b2^") * Fock([1], [], [])
    assert out.op_dict == {((0, 1, 1), (0, 2, 1)): -1.0}


def op_on_state_3():
    out = ParticleOperator("a0^ a0^ a0^") * Fock([1], [], [])
    assert out.op_dict == {((0, 1, 1), (2, 0, 1), (2, 0, 1), (2, 0, 1)): np.sqrt(6)}


def op_on_state_4():
    out = (ParticleOperator("b1^") + ParticleOperator("a1^")) * Fock([], [], [])
    assert out.op_dict == {((0, 1, 1),): 1.0, ((2, 1, 1),): 1.0}


def test_defined_matrix_elem_fock_sum_1():
    bra = Fock([0], [1], []) + Fock([0], [2], [])
    operator = ParticleOperator("b0^")
    state = Fock([], [1], []) + Fock([], [2], [])
    me = get_matrix_element(bra, state, operator)
    assert me == 2.0

from openparticle import (
    ParticleOperator,
    Fock,
)
import numpy as np
from openparticle.utils import get_matrix_element, overlap, generate_matrix
import pytest


def test_op_on_state_1():
    out = ParticleOperator("b0^") * Fock([1], [], [])
    assert out.state_dict == {((0, 1), (), ()): 1.0}


def test_op_on_state_2():
    out = ParticleOperator("b1^") * Fock([0], [], [])
    assert out.state_dict == {((0, 1), (), ()): -1.0}


def test_op_on_state_3():
    out = ParticleOperator("a0^ a0^ a0^") * Fock([1], [], [])
    assert out.state_dict == {((1,), (), ((0, 3),)): np.sqrt(6)}


def test_op_on_state_4():
    out = (ParticleOperator("b1^") + ParticleOperator("a1^")) * Fock([], [], [])
    assert out.state_dict == {((1,), (), ()): 1.0, ((), (), ((1, 1),)): 1.0}


def test_defined_matrix_elem_fock_sum_1():
    bra = Fock([0], [1], []) + Fock([0], [2], [])
    operator = ParticleOperator("b0^")
    state = Fock([], [1], []) + Fock([], [2], [])
    me = get_matrix_element(bra, operator, state)
    assert me == 2.0


def test_defined_matrix_elem_in_particleoperator_sum_and_fock_sum_class_2():
    bra = Fock([0], [1], []) + Fock([1], [3], []) + Fock([1], [3], [])
    operator = (
        ParticleOperator("b0^") + ParticleOperator("b1^") + ParticleOperator("b1^")
    )
    state = Fock([], [1], []) + Fock([], [3], []) + Fock([], [3], [])

    assert get_matrix_element(bra, operator, state) == 9.0


def test_defined_matrix_elem_in_particleoperator_sum_and_fock_sum_class_multi_1():
    bra = (
        Fock([0], [1], [])
        + Fock([1], [2], [])
        + Fock([2], [3], [])
        + Fock([3], [4], [])
        + Fock([4], [5], [])
    )
    operator = (
        ParticleOperator("b0^")
        + ParticleOperator("b1^")
        + ParticleOperator("b2^")
        + ParticleOperator("b3^")
        + ParticleOperator("b4^")
    )
    state = (
        Fock([], [1], [])
        + Fock([], [2], [])
        + Fock([], [3], [])
        + Fock([], [4], [])
        + Fock([], [5], [])
    )

    assert get_matrix_element(bra, operator, state) == 5.0


def test_defined_matrix_elem_in_particleoperator_sum_and_fock_sum_class_multi_2():
    bra = (
        Fock([0], [1], [])
        + Fock([0], [3], [])
        + Fock([0], [3], [])
        + Fock([0], [3], [])
        + Fock([0], [3], [])
    )
    operator = ParticleOperator("b0^") + ParticleOperator("b0^")
    state = Fock([], [1], []) + Fock([], [3], [])

    assert get_matrix_element(bra, operator, state) == 10.0


def test_defined_matrix_elem_in_particleoperator_sum_and_fock_sum_class_multi_3():
    bra = (
        Fock([0], [1], [])
        + Fock([0], [3], [])
        + Fock([0], [3], [])
        + Fock([0], [3], [])
        + Fock([1], [3], [])
    )
    operator = (
        ParticleOperator("b0^") + ParticleOperator("b0^") + ParticleOperator("b1^")
    )
    state = Fock([], [1], []) + Fock([], [3], [])

    assert get_matrix_element(bra, operator, state) == 9.0


def test_identity_on_state():
    identity = ParticleOperator(" ")
    state = Fock([1], [0, 1, 4], [(0, 3)])
    output = identity * state
    assert output.state_dict == state.state_dict


def test_identity_in_sum_on_state():
    op = ParticleOperator(" ") + ParticleOperator("b1^ b1")
    state = Fock([1], [0, 1, 4], [(0, 3)])
    output = op * state
    assert output.state_dict == {((1,), (0, 1, 4), ((0, 3),)): 2.0}


def test_fermionic_parity_1():
    state = Fock([0, 1], [], [])
    op = ParticleOperator("b0^ b0")
    out = op * state
    assert out.state_dict == state.state_dict


def test_fermionic_parity_2():
    state = Fock([0, 1], [], [])
    op = ParticleOperator("b0^ b0") + ParticleOperator("b1^ b1")
    out = op * state
    expected_out = 2 * state
    assert out.state_dict == expected_out.state_dict

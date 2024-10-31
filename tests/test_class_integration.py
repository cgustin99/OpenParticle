from openparticle import (
    ParticleOperator,
    Fock,
)
import numpy as np
from openparticle.utils import get_matrix_element, overlap, generate_matrix
import pytest
import math
from collections import Counter


def test_fock_state_generation():
    # |1;1> = b_1^ d_1^ |vac>
    state = Fock([1], [1], [])
    op_that_creates_state = ParticleOperator("b1^ d1^")
    assert op_that_creates_state.act_on_vacuum().op_dict == state.op_dict


def test_fock_state_generation_2():
    # d1^ b1^ |vac> = -|1;1>
    assert (
        ParticleOperator("d1^ b1^").act_on_vacuum().op_dict
        == (-1 * Fock([1], [1], [])).op_dict
    )


def test_fock_state_generation_3():
    assert ParticleOperator("b1^ d1^").act_on_vacuum().op_dict == {
        ((0, 1, 1), (1, 1, 1)): 1.0
    }


def test_fock_state_generation_4():
    assert ParticleOperator("d1^ b1^").act_on_vacuum().op_dict == {
        ((0, 1, 1), (1, 1, 1)): -1.0
    }


def test_fock_state_generation_5():
    assert ParticleOperator("b1^ b2^").act_on_vacuum().op_dict == {
        ((0, 1, 1), (0, 2, 1)): 1.0
    }


def test_fock_state_generation_6():
    assert ParticleOperator("b2^ b1^").act_on_vacuum().op_dict == {
        ((0, 1, 1), (0, 2, 1)): -1.0
    }


def test_bosonic_fock_state_generation():
    state = Fock([], [], [(0, 3)])
    op = 1 / (np.sqrt(math.factorial(3))) * ParticleOperator("a0^ a0^ a0^")
    vacuum = Fock([], [], [])

    assert (op * vacuum).coeff == state.coeff


def test_bosonic_fock_state_generation_2():
    vacuum = Fock([], [], [])
    op = ParticleOperator("a0^ a1^")
    expected_state = Fock([], [], [[0, 1], [1, 1]])
    assert (op * vacuum).coeff == expected_state.coeff  # == 1


def test_bosonic_fock_state_generation_3():
    vacuum = Fock([], [], [])
    op = 1 / np.sqrt(2) * ParticleOperator("a0^ a0^ a1^")
    expected_state = Fock([], [], [[0, 2], [1, 1]])
    assert (op * vacuum).coeff == expected_state.coeff  # np.sqrt(2)


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


@pytest.mark.parametrize("trial", range(10))
def test_fock_to_particleop_back_to_fock(trial):
    # |state> = op_for_state|vac> = |state>

    rand_length = np.random.randint(10)
    f_occ = sorted(set(np.random.randint(0, 10, size=rand_length)))
    af_occ = sorted(set(np.random.randint(0, 10, size=rand_length)))

    counted = Counter(np.random.randint(0, 10, size=rand_length))
    result = [(num, count) for num, count in counted.items()]
    b_occ = sorted(result)

    state = Fock(f_occ, af_occ, b_occ)
    assert (
        ParticleOperator(state.op_dict).act_on_vacuum().state_dict == state.state_dict
    )


def test_state_annihilation_1():
    assert ParticleOperator("b0^ b0^").act_on_vacuum() == 0


def test_state_annihilation_2():
    assert ParticleOperator("d0").act_on_vacuum() == 0


def test_state_annihilation_3():
    assert ParticleOperator("d0 d0 b0^").act_on_vacuum() == 0


def test_state_annihilation_4():
    assert ParticleOperator("b0").act_on_vacuum() == 0


def test_state_annihilation_5():
    assert ParticleOperator("a0").act_on_vacuum() == 0


def test_state_annihilation_6():
    assert ParticleOperator("a0 a0") * Fock([], [], [(0, 1)]) == 0

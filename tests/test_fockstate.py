from openparticle import Fock, ParticleOperator
import pytest
import numpy as np
import math


def test_fock_state_stores_correct_occupancies():
    f_occ = [2]  # One fermion in mode 2
    af_occ = [1, 3]  # One antifermion in mode 1, one in mode 3
    b_occ = [(0, 3)]  # Three bosons in mode 0

    fs = Fock(f_occ=f_occ, af_occ=af_occ, b_occ=b_occ)
    assert list(next(iter(fs.state_dict)))[0] == tuple(f_occ)
    assert list(next(iter(fs.state_dict)))[1] == tuple(af_occ)
    assert list(next(iter(fs.state_dict)))[2] == tuple(b_occ)


@pytest.mark.parametrize("coeff", np.random.uniform(-100, 100, size=3))
def test_fock_state_mul_by_constant(coeff):
    f_occ = [2]  # One fermion in mode 2
    af_occ = [1, 3]  # One antifermion in mode 1, one in mode 3
    b_occ = [(0, 3)]  # Three bosons in mode 0

    fs = Fock(f_occ=f_occ, af_occ=af_occ, b_occ=b_occ)

    fs_new = coeff * fs
    assert fs_new.coeff == coeff


def test_fock_states_add_two_different_states():
    state = Fock([1, 2], [], []) + Fock([], [1], [])
    assert state.op_dict == {((0, 1, 1), (0, 2, 1)): 1.0, ((1, 1, 1),): 1.0}


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

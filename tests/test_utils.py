from openparticle import (
    ParticleOperator,
    Fock,
)
import numpy as np
from openparticle.utils import get_matrix_element, overlap, generate_matrix
import pytest


def test_overlap_1():
    bra_state = Fock([1, 2], [3], [(0, 1)])
    ket_state = Fock([1, 2], [3], [(0, 1)])
    assert overlap(bra_state, ket_state) == 1.0


def test_overlap_2():
    bra_state = Fock([1, 2], [3], [(0, 1)]) + Fock([1], [], [])
    ket_state = Fock([1, 2], [3], [(0, 1)])
    assert overlap(bra_state, ket_state) == 1.0


def test_overlap_3():
    bra_state = Fock([1, 2], [3], [(0, 1)]) + Fock([1], [], [])
    ket_state = Fock([1, 3], [3], [(0, 1)])
    assert overlap(bra_state, ket_state) == 0


def test_matrix_generation_1():
    basis = [
        Fock([], [], []),
        Fock([0], [], []),
        Fock([1], [], []),
        Fock([0, 1], [], []),
    ]
    operator = (
        ParticleOperator("b0^ b0")
        + ParticleOperator("b1^ b0")
        + ParticleOperator("b0^ b1")
        + ParticleOperator("b1^ b1")
    )

    assert np.allclose(
        generate_matrix(operator, basis),
        np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 2]]),
    )


def test_matrix_generation_2():
    basis = [
        Fock([], [], []),
        Fock([], [], [[0, 1]]),
        Fock([], [], [[1, 1]]),
        Fock([], [], [[0, 1], [1, 1]]),
    ]
    operator = (
        ParticleOperator("a0^ a0")
        + ParticleOperator("a1^ a0")
        + ParticleOperator("a0^ a1")
        + ParticleOperator("a1^ a1")
    )

    assert np.allclose(
        generate_matrix(operator, basis),
        np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 2]]),
    )


def test_matrix_generation_3():
    basis = [
        Fock([], [], []),
        Fock([], [], [[0, 1]]),
        Fock([], [], [[1, 1]]),
        Fock([], [], [[0, 1], [1, 1]]),
    ]
    op = (
        ParticleOperator("a0")
        + ParticleOperator("a1")
        + ParticleOperator("a0^ a1")
        + ParticleOperator("a1^ a0")
    )
    expected = np.array(
        [
            [0.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ]
    )
    assert np.allclose(expected, generate_matrix(op, basis))

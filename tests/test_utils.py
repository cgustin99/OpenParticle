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


def test_overlap_4():
    assert np.allclose(overlap(Fock([], [], [[0, 2]]), Fock([], [], [[0, 2]])), 1.0)


def test_VEV_1():
    assert ParticleOperator("a0 a0 a0^ a0^").VEV() == 2.0


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


def test_matrix_generation_4():
    op = (
        ParticleOperator("d0^ a0^ b0")
        - 0.5 * ParticleOperator("b0^ d0")
        + 0.25 * ParticleOperator("b0^ d0^ a0")
        + 1 / 3 * ParticleOperator("a0^ d0")
    )
    basis = [
        Fock.vacuum(),
        Fock([], [], [[0, 1]]),
        Fock([], [0], []),
        Fock([], [0], [[0, 1]]),
        Fock([0], [], []),
        Fock([0], [], [[0, 1]]),
        Fock([0], [0], []),
        Fock([0], [0], [[0, 1]]),
    ]

    expected = np.array(
        [
            [
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                7.85046229e-17 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
            ],
            [
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                3.33333333e-01 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
            ],
            [
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                -1.57009246e-16 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
            ],
            [
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                1.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
            ],
            [
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                -5.00000000e-01 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                7.85046229e-17 + 0.0j,
            ],
            [
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                -5.00000000e-01 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                -3.33333333e-01 + 0.0j,
                0.00000000e00 + 0.0j,
            ],
            [
                0.00000000e00 + 0.0j,
                2.50000000e-01 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
            ],
            [
                -3.92523115e-17 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
            ],
        ]
    )

    assert np.allclose(generate_matrix(op, basis), expected)


def test_matrix_generation_5():
    op = ParticleOperator("a0")
    basis = [
        Fock.vacuum(),
        Fock([], [], [[0, 1]]),
        Fock([], [], [[0, 2]]),
        Fock([], [], [[0, 3]]),
    ]

    expected = np.array(
        [
            [
                0.00000000e00 + 0.0j,
                1.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
            ],
            [
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                1.41421356e00 + 0.0j,
                0.00000000e00 + 0.0j,
            ],
            [
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                1.73205081e00 + 0.0j,
            ],
            [
                -1.57009246e-16 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
                0.00000000e00 + 0.0j,
            ],
        ]
    )
    assert np.allclose(generate_matrix(op, basis), expected)

from openparticle import (
    ParticleOperator,
    Fock,
    ConjugateFock,
)
import numpy as np
import pytest


def test_defined_matrix_element_1():
    bra = ConjugateFock([1], [], [])
    operator = ParticleOperator("b1^")
    ket = Fock([], [], [])

    assert bra * operator * ket == 1.0


def test_defined_matrix_element_2():
    bra = ConjugateFock([1], [], [])
    operator = ParticleOperator("b0^")
    ket = Fock([], [], [])

    assert bra * operator * ket == 0


def test_defined_particleoperator_acts_on_fock_state():
    operator = ParticleOperator("b0^")
    state = Fock([], [1], [])

    assert (operator * state).state_dict == (Fock([0], [1], [])).state_dict


def test_defined_particleoperatorsum_act_on_matrix_element_1():
    bra = ConjugateFock([1], [], [])
    operator = ParticleOperator("b1^") + ParticleOperator("b1^")
    ket = Fock([], [], [])

    assert bra * (Fock([1], [], []) + Fock([1], [], [])) == 2.0
    assert (operator * ket).state_dict == (
        Fock([1], [], []) + Fock([1], [], [])
    ).state_dict
    assert bra * operator * ket == 2.0


def test_defined_particleoperatorsum_act_on_matrix_element_2():
    bra = ConjugateFock([1], [], [(1, 1)])
    operator = ParticleOperator("b1^") + ParticleOperator("a1^")
    ket = Fock([], [], [])

    assert bra * operator * ket == 0


def test_defined_particleoperator_acts_on_fock_sum_diff():
    operator = ParticleOperator("b0^")
    state = Fock([], [1], []) + Fock([], [2], [])

    assert (operator * state).state_dict == {
        ((0,), (1,), ()): 1,
        ((0,), (2,), ()): 1,
    }


def test_defined_particleoperator_acts_on_fock_sum_same():
    operator = ParticleOperator("b0^")
    state = Fock([], [1], []) + Fock([], [1], [])

    assert (operator * state).state_dict == {((0,), (1,), ()): 2.0}


def test_defined_matrix_elem_fock_sum_1():
    bra = ConjugateFock([0], [1], []) + ConjugateFock([0], [2], [])
    operator = ParticleOperator("b0^")
    state = Fock([], [1], []) + Fock([], [2], [])

    assert bra * operator * state == 2.0


def test_defined_matrix_elem_fock_sum_2():
    bra = ConjugateFock([0], [1], [])
    operator = ParticleOperator("b0^")
    state = Fock([], [1], []) + Fock([], [1], [])

    assert bra * operator * state == 2.0


def test_defined_matrix_elem_fock_sum_diff_1():
    bra = ConjugateFock([0], [], [])
    operator = ParticleOperator("b0^")
    state = Fock([], [1], []) + Fock([], [1], [])

    assert bra * operator * state == 0


def test_defined_matrix_elem_fock_sum_diff_2():
    bra = ConjugateFock([0], [1], [])
    operator = ParticleOperator("b0^")
    state = Fock([], [1], []) + Fock([], [2], [])

    assert bra * operator * state == 1.0


def test_defined_conj_fock_sum_same():
    bra = ConjugateFock([0], [1], []) + ConjugateFock([0], [1], [])
    operator = ParticleOperator("b0^")
    state = Fock([], [1], [])

    assert bra * operator * state == 2.0


def test_defined_conj_fock_sum_diff_1():
    bra = ConjugateFock([0], [1], []) + ConjugateFock([0], [2], [])
    operator = ParticleOperator("b0^")
    state = Fock([], [1], [])

    assert bra * operator * state == 1.0


def test_defined_conj_fock_sum_diff_2():
    bra = ConjugateFock([0], [1], []) + ConjugateFock([0], [1], [])
    operator = ParticleOperator("b0^")
    state = Fock([], [], [])

    assert bra * operator * state == 0


def test_defined_matrix_elem_in_sum_class_1():
    bra = ConjugateFock([0], [1], []) + ConjugateFock([0], [3], [])
    operator = ParticleOperator("b0^")
    state = Fock([], [1], []) + Fock([], [3], [])

    assert bra * operator * state == 2.0


def test_defined_matrix_elem_in_sum_class_2():
    bra = ConjugateFock([1], [1], []) + ConjugateFock([0], [3], [])
    operator = ParticleOperator("b0^")
    state = Fock([], [1], []) + Fock([], [3], [])

    assert bra * operator * state == 1.0


def test_defined_matrix_elem_in_sum_class_3():
    bra = ConjugateFock([1], [1], []) + ConjugateFock([1], [3], [])
    operator = ParticleOperator("b0^")
    state = Fock([], [1], []) + Fock([], [3], [])

    assert bra * operator * state == 0


def test_defined_matrix_elem_in_fock_sum_class_multi_1():
    bra = (
        ConjugateFock([0], [1], [])
        + ConjugateFock([0], [2], [])
        + ConjugateFock([0], [3], [])
        + ConjugateFock([0], [4], [])
        + ConjugateFock([0], [5], [])
    )
    operator = ParticleOperator("b0^")
    state = (
        Fock([], [1], [])
        + Fock([], [2], [])
        + Fock([], [3], [])
        + Fock([], [4], [])
        + Fock([], [5], [])
    )

    assert bra * operator * state == 5.0


def test_defined_matrix_elem_in_fock_sum_class_multi_2():
    bra = (
        ConjugateFock([0], [1], [])
        + ConjugateFock([0], [3], [])
        + ConjugateFock([0], [3], [])
        + ConjugateFock([0], [3], [])
        + ConjugateFock([0], [3], [])
    )
    operator = ParticleOperator("b0^")
    state = Fock([], [1], []) + Fock([], [3], [])

    assert bra * operator * state == 5.0


def test_defined_matrix_elem_in_particleoperator_sum_and_fock_sum_class_1():
    bra = ConjugateFock([0], [1], []) + ConjugateFock([1], [3], [])
    operator = ParticleOperator("b0^") + ParticleOperator("b1^")
    state = Fock([], [1], []) + Fock([], [3], [])

    assert bra * operator * state == 2.0


def test_defined_matrix_elem_in_particleoperator_sum_and_fock_sum_class_2():
    bra = (
        ConjugateFock([0], [1], [])
        + ConjugateFock([1], [3], [])
        + ConjugateFock([1], [3], [])
    )
    operator = (
        ParticleOperator("b0^") + ParticleOperator("b1^") + ParticleOperator("b1^")
    )
    state = Fock([], [1], []) + Fock([], [3], []) + Fock([], [3], [])

    assert bra * operator * state == 9.0


def test_defined_matrix_elem_in_particleoperator_sum_and_fock_sum_class_multi_1():
    bra = (
        ConjugateFock([0], [1], [])
        + ConjugateFock([1], [2], [])
        + ConjugateFock([2], [3], [])
        + ConjugateFock([3], [4], [])
        + ConjugateFock([4], [5], [])
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

    assert bra * operator * state == 5.0


def test_defined_matrix_elem_in_particleoperator_sum_and_fock_sum_class_multi_2():
    bra = (
        ConjugateFock([0], [1], [])
        + ConjugateFock([0], [3], [])
        + ConjugateFock([0], [3], [])
        + ConjugateFock([0], [3], [])
        + ConjugateFock([0], [3], [])
    )
    operator = ParticleOperator("b0^") + ParticleOperator("b0^")
    state = Fock([], [1], []) + Fock([], [3], [])

    assert bra * operator * state == 10.0


def test_defined_matrix_elem_in_particleoperator_sum_and_fock_sum_class_multi_3():
    bra = (
        ConjugateFock([0], [1], [])
        + ConjugateFock([0], [3], [])
        + ConjugateFock([0], [3], [])
        + ConjugateFock([0], [3], [])
        + ConjugateFock([1], [3], [])
    )
    operator = (
        ParticleOperator("b0^") + ParticleOperator("b0^") + ParticleOperator("b1^")
    )
    state = Fock([], [1], []) + Fock([], [3], [])

    assert bra * operator * state == 9.0


def test_bra_times_bra():
    bra = ConjugateFock([0], [1], [])

    assert bra * bra is None


def test_bra_sum_times_bra_sum_same():
    bra_sum = ConjugateFock([0], [1], []) + ConjugateFock([0], [1], [])

    assert bra_sum * bra_sum is None


def test_bra_sum_times_bra_sum_diff():
    bra_sum = ConjugateFock([0], [1], []) + ConjugateFock([0], [2], [])

    assert bra_sum * bra_sum is None

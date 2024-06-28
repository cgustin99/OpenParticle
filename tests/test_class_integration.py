from openparticle import (
    ParticleOperator,
    ParticleOperatorSum,
    Fock,
    FockSum,
    ConjugateFock,
    ConjugateFockSum,
)
import numpy as np
import pytest


def test_defined_matrix_element():
    bra = ConjugateFock([1], [], [])
    operator = ParticleOperator("b1^")
    ket = Fock([], [], [])

    assert bra * operator * ket == 1.0


def test_defined_particleoperator_acts_on_fock_state():
    operator = ParticleOperator("b0^")
    state = Fock([], [1], [])

    assert operator * state == Fock([0], [1], [])


def test_random_particleoperator_acts_on_fock_state():
    pass

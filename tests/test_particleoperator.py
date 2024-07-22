from openparticle import ParticleOperator
import pytest
import numpy as np


def test_particleoperator_generates_correctly():
    input_str = "b0^ b2 a3"
    particleop = ParticleOperator(input_str)

    assert particleop.op_dict.popitem()[0] == input_str


@pytest.mark.parametrize("coeff", np.random.uniform(-100, 100, size=10))
def test_coeoff_times_particleoperator(coeff):
    input_str = "b0^ b2 a3"
    particleop = coeff * ParticleOperator(input_str)

    assert next(iter(particleop.op_dict.values())) == coeff


def test_add_two_different_particleoperators():
    op1 = ParticleOperator("b0")
    op2 = ParticleOperator("b2")

    op = op1 + op2
    opsum = ParticleOperator({"b0": 1, "b2": 1})
    assert opsum.op_dict == op.op_dict


def test_add_two_same_particleoperators():
    op1 = ParticleOperator("b0")
    op2 = ParticleOperator("b0")

    assert (op1 + op2).op_dict == {"b0": 2}


def test_add_particleoperatorsum_to_particleoperator():
    po = ParticleOperator("b0")
    pos = ParticleOperator({"b1": 1, "d1": 1})
    out = ParticleOperator({"b1": 1, "d1": 1, "b0": 1})

    assert (po + pos).op_dict == out.op_dict


def test_add_particleoperatorsums():
    pos1 = ParticleOperator({"b1": 1, "d1": 1})
    pos2 = ParticleOperator({"a1": 1, "a2": 1})

    assert (pos1 + pos2).op_dict == {"b1": 1, "d1": 1, "a1": 1, "a2": 1}


@pytest.mark.parametrize("power", np.arange(1, 10, 1))
def test_particle_operator_to_some_power(power):
    operator = ParticleOperator("a0")
    assert (operator**power).op_dict == (
        ParticleOperator((("a0" + " ") * power)[:-1])
    ).op_dict

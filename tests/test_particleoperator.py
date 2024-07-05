from openparticle import ParticleOperator, ParticleOperatorSum
import pytest
import numpy as np


def test_particleoperator_generates_correctly():
    input_str = "b0^ b2 a3"
    particleop = ParticleOperator(input_str)

    assert particleop.input_string == input_str


@pytest.mark.parametrize("coeff", np.random.uniform(-100, 100, size=10))
def test_coeoff_times_particleoperator(coeff):
    input_str = "b0^ b2 a3"
    particleop = coeff * ParticleOperator(input_str)

    assert particleop.coeff == coeff


def test_add_two_different_particleoperators():
    op1 = ParticleOperator("b0")
    op2 = ParticleOperator("b2")

    opsum = op1 + op2

    assert opsum.__str__() == "1.0 * b0 + 1.0 * b2"


def test_add_two_same_particleoperators():
    op1 = ParticleOperator("b0")
    op2 = ParticleOperator("b0")

    assert (op1 + op2).__str__() == "2.0 * b0"


def test_add_particleoperatorsum_to_particleoperator():
    po = ParticleOperator("b0")
    pos = ParticleOperator("b1") + ParticleOperator("d1")

    assert (po + pos).__str__() == "1.0 * b1 + 1.0 * d1 + 1.0 * b0"


def test_add_particleoperator_to_particleoperatorsum():
    po = ParticleOperator("b0")
    pos = ParticleOperator("b1") + ParticleOperator("d1")

    assert (po + pos).__str__() == "1.0 * b1 + 1.0 * d1 + 1.0 * b0"


def test_add_particleoperatorsums():
    pos1 = ParticleOperator("b1") + ParticleOperator("d1")
    pos2 = ParticleOperator("a1") + ParticleOperator("a2")

    assert (pos1 + pos2).__str__() == "1.0 * b1 + 1.0 * d1 + 1.0 * a1 + 1.0 * a2"

def test_add_overlapping_particleoperatorsums():
    pos1 = ParticleOperator("b1") + ParticleOperator("d1")
    pos2 = ParticleOperator("b1") + ParticleOperator("a2")

    assert (pos1 + pos2).__str__() == "2.0 * b1 + 1.0 * d1 + 1.0 * a2"

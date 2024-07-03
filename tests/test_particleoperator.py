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

    assert opsum == ParticleOperatorSum([op1, op2])


def test_add_two_same_particleoperators():
    op1 = ParticleOperator("b0")
    op2 = ParticleOperator("b0")

    assert op1 + op2 == 2 * ParticleOperator("b0")


def test_add_particleoperatorsum_to_particleoperator():
    po = ParticleOperator("b0")
    pos = ParticleOperatorSum([ParticleOperator("b1"), ParticleOperator("d1")])

    assert po + pos == ParticleOperatorSum(
        [po, pos.operator_list[0], pos.operator_list[1]]
    )


def test_add_particleoperator_to_particleoperatorsum():
    po = ParticleOperator("b0")
    pos = ParticleOperatorSum([ParticleOperator("b1"), ParticleOperator("d1")])

    assert pos + po == ParticleOperatorSum(
        [po, pos.operator_list[0], pos.operator_list[1]]
    )


def test_add_particleoperatorsums():
    pos1 = ParticleOperatorSum([ParticleOperator("b1"), ParticleOperator("d1")])
    pos2 = ParticleOperatorSum([ParticleOperator("a1"), ParticleOperator("a2")])

    assert pos1 + pos2 == ParticleOperatorSum(
        [
            pos1.operator_list[0],
            pos1.operator_list[1],
            pos2.operator_list[0],
            pos2.operator_list[1],
        ]
    )


def test_add_overlapping_particleoperatorsums():
    pos1 = ParticleOperatorSum([ParticleOperator("b1"), ParticleOperator("d1")])
    pos2 = ParticleOperatorSum([ParticleOperator("b1"), ParticleOperator("a2")])

    assert pos1 + pos2 == 2 * ParticleOperator("b1") + ParticleOperator(
        "d1"
    ) + ParticleOperator("a2")

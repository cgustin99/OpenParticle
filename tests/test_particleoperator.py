from openparticle import (
    ParticleOperator,
    OccupationOperator,
    BosonOperator,
    FermionOperator,
)
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


@pytest.mark.parametrize("coeff", np.random.uniform(-100, 100, size=10))
def test_coeff_property_on_singleton(coeff):
    op = ParticleOperator({"a0^ a1": coeff})
    assert op.coeff == coeff


def test_coeff_property_on_larger_dictionary():
    coeffs = [1, 2, 3]
    op = (
        coeffs[0] * ParticleOperator("a0^ a1")
        + coeffs[1] * ParticleOperator("b1^ d2")
        + coeffs[2] * ParticleOperator("a0^ a0^ a0")
    )
    assert op.coeffs == coeffs


def test_particle_operator_identity_product():
    op1 = ParticleOperator("a0^") ** 0
    op2 = ParticleOperator("a0") ** 0
    op = op1 * op2
    assert op.op_dict == {" ": 1}


def test_identity_dagger():
    identity = ParticleOperator(" ")
    dagger = identity.dagger()
    assert identity.op_dict == dagger.op_dict


def test_power_with_multiple_terms_in_dict_1():
    power = 2
    op = ParticleOperator("a0^") + ParticleOperator("a0")
    output = op**power
    expected = (
        ParticleOperator("a0^ a0^")
        + ParticleOperator("a0^ a0")
        + ParticleOperator("a0 a0^")
        + ParticleOperator("a0 a0")
    )

    assert output.op_dict == expected.op_dict


def test_power_with_multiple_terms_in_dict_2():
    power = 3
    op = ParticleOperator("a0^") + ParticleOperator("a0")
    output = op**power
    expected = (
        ParticleOperator("a0^ a0^ a0^")
        + ParticleOperator("a0^ a0^ a0")
        + ParticleOperator("a0^ a0 a0^")
        + ParticleOperator("a0^ a0 a0")
        + ParticleOperator("a0 a0^ a0^")
        + ParticleOperator("a0 a0^ a0")
        + ParticleOperator("a0 a0 a0^")
        + ParticleOperator("a0 a0 a0")
    )

    assert output.op_dict == expected.op_dict


def test_power_with_multiple_terms_in_dict_3():
    power = 4
    op = ParticleOperator("a0^") + ParticleOperator("a0")
    output = op**power
    expected = (
        ParticleOperator("a0 a0 a0 a0")
        + ParticleOperator("a0 a0 a0 a0^")
        + ParticleOperator("a0 a0 a0^ a0")
        + ParticleOperator("a0 a0 a0^ a0^")
        + ParticleOperator("a0 a0^ a0 a0")
        + ParticleOperator("a0 a0^ a0 a0^")
        + ParticleOperator("a0 a0^ a0^ a0")
        + ParticleOperator("a0 a0^ a0^ a0^")
        + ParticleOperator("a0^ a0 a0 a0")
        + ParticleOperator("a0^ a0 a0 a0^")
        + ParticleOperator("a0^ a0 a0^ a0")
        + ParticleOperator("a0^ a0 a0^ a0^")
        + ParticleOperator("a0^ a0^ a0 a0")
        + ParticleOperator("a0^ a0^ a0 a0^")
        + ParticleOperator("a0^ a0^ a0^ a0")
        + ParticleOperator("a0^ a0^ a0^ a0^")
    )
    assert output.op_dict == expected.op_dict


def test_len_magic_method_for_particle_op():
    length = 3
    op = ParticleOperator("a0") + ParticleOperator("b0") + ParticleOperator("d0")
    assert len(op) == length


def test_has_fermions():
    op = ParticleOperator("b0^ b1") + ParticleOperator("a0")
    assert op.has_fermions


def test_has_fermions_2():
    op = ParticleOperator("a0^ a0") + ParticleOperator("a0") + ParticleOperator("d1")
    assert op.has_fermions == False


def test_has_antifermions():
    op = ParticleOperator("b0^ b1") + ParticleOperator("a0")
    assert op.has_antifermions == False


def test_has_antifermions_2():
    op = ParticleOperator("a0^ a0") + ParticleOperator("a0") + ParticleOperator("d1")
    assert op.has_antifermions


def test_has_bosons():
    op = ParticleOperator("b0^ b1") + ParticleOperator("a0")
    assert op.has_bosons


def test_has_bosons2():
    op = ParticleOperator("b1") + ParticleOperator("b0") + ParticleOperator("d1")
    assert op.has_bosons == False


def test_parse_fermions():
    op = ParticleOperator("b0^ b0 b1^ b1")
    expected_ops = [OccupationOperator("b", 0, 1), OccupationOperator("b", 1, 1)]
    # assert op.parse() ==
    for parsed_op, expected_op in zip(op.parse(), expected_ops):
        assert str(parsed_op) == str(expected_op)


def test_parse_antifermions():
    op = ParticleOperator("d1^ d1 d0^ d0")
    expected_ops = [OccupationOperator("d", 1, 1), OccupationOperator("d", 0, 1)]
    for parsed_op, expected_op in zip(op.parse(), expected_ops):
        assert str(parsed_op) == str(expected_op)


def test_parse_bosons():
    op = ParticleOperator("a1^ a1 a2^ a2^ a2^ a2 a2 a1^ a1 a0")
    expected_ops = [
        OccupationOperator("a", 1, 1),
        BosonOperator("2^"),
        OccupationOperator("a", 2, 2),
        OccupationOperator("a", 1, 1),
        BosonOperator("0"),
    ]
    for parsed_op, expected_op in zip(op.parse(), expected_ops):
        assert str(parsed_op) == str(expected_op)


def test_parse_mix():
    op = ParticleOperator("a0^ a0 a1^ a1^ a1^ a1 a1 b0^ d0^ d0 a0")
    expected_ops = [
        OccupationOperator("a", 0, 1),
        BosonOperator("1^"),
        OccupationOperator("a", 1, 2),
        FermionOperator("0^"),
        OccupationOperator("d", 0, 1),
        BosonOperator("0"),
    ]
    for parsed_op, expected_op in zip(op.parse(), expected_ops):
        assert str(parsed_op) == str(expected_op)

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

    assert (
        ParticleOperator.key_to_op_string(particleop.op_dict.popitem()[0]) == input_str
    )


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

    assert (op1 + op2).op_dict == ParticleOperator({"b0": 2}).op_dict


def test_add_different_initializations_of_particleoperators():
    po = ParticleOperator("b0") + ParticleOperator("a2")
    pos = ParticleOperator({"b1": 1, "d1": 1})
    out = ParticleOperator({"b1": 1, "d1": 1, "b0": 1, "a2": 1})

    assert (po + pos).op_dict == out.op_dict


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
    assert op.op_dict == {(): 1}


def test_identity_dagger():
    identity = ParticleOperator(" ")
    dagger = identity.dagger()
    assert identity.op_dict == dagger.op_dict


def test_conjugate_particle_operator():
    op = ParticleOperator("b0 d1^")
    conjugated_op = ParticleOperator("d1 b0^")
    assert conjugated_op.op_dict == op.dagger().op_dict


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


def test_max_mode():
    modes = np.random.randint(0, 100, size=3)
    op_str = (
        "b" + str(modes[0]) + "^ " + "d" + str(modes[1]) + "^ " + "a" + str(modes[2])
    )

    assert ParticleOperator(op_str).max_mode() == max(modes)


def test_max_mode_2():
    modes = np.random.randint(0, 100, size=4)
    op = ParticleOperator(
        "b" + str(modes[0]) + "^ " + "d" + str(modes[1])
    ) + ParticleOperator("a" + str(modes[2]) + "^ " + "a" + str(modes[3]))
    assert op.max_mode() == max(modes)


def test_parse_fermions():
    op = ParticleOperator("b0^ b0 b1^ b1")
    expected_ops = [OccupationOperator("b", 0, 1), OccupationOperator("b", 1, 1)]
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


def test_parse_bosons2():
    op = ParticleOperator("a1^ a1 a1 a1")
    expected_ops = [
        OccupationOperator("a", 1, 1),
        BosonOperator("1"),
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


# def test_random_particleoperator_generation():
#     for _ in range(1000):
#         possible_types = [
#             ["fermion"],
#             ["antifermion"],
#             ["boson"],
#             ["fermion", "antifermion"],
#             ["fermion", "boson"],
#             ["antifermion", "boson"],
#             ["fermion", "antifermion", "boson"],
#         ]
#         types = possible_types[np.random.choice(range(7))]
#         n_terms = np.random.choice(range(1, 17))
#         max_mode = np.random.choice(range(1, 9))
#         max_len_of_terms = np.random.choice(range(1, 9))
#         operator = ParticleOperator.random(
#             types,
#             n_terms,
#             max_mode=max_mode,
#             max_len_of_terms=max_len_of_terms,
#             complex_coeffs=False,
#             normal_order=True,
#         ).normal_order()

#         assert isinstance(operator, ParticleOperator)

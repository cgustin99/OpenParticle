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

    assert ParticleOperator(op_str).max_mode == max(modes)


def test_max_mode_2():
    modes = np.random.randint(0, 100, size=4)
    op = ParticleOperator(
        "b" + str(modes[0]) + "^ " + "d" + str(modes[1])
    ) + ParticleOperator("a" + str(modes[2]) + "^ " + "a" + str(modes[3]))
    assert op.max_mode == max(modes)


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


def test_split():
    op = ParticleOperator("a0^ a0")
    split_op = op.split()
    assert split_op[0].op_dict == BosonOperator("0^").op_dict
    assert split_op[1].op_dict == BosonOperator("0").op_dict


def test_equality1():
    op1 = ParticleOperator("b0^ d1")
    op2 = ParticleOperator("b0^ d1")
    return op1 == op2


def test_equality2():
    op1 = ParticleOperator("b1^ d1")
    op2 = ParticleOperator("b0^ d1")
    return op1 != op2


def test_equality3():
    op1 = ParticleOperator("b10^ d1") + ParticleOperator("a0^")
    op2 = ParticleOperator("b0^ d1")
    return op1 != op2


def test_equality4():
    op1 = ParticleOperator("b10^ d1") + ParticleOperator("a0^")
    op2 = ParticleOperator("b10^ d1") + ParticleOperator("a0^")
    return op1 == op2


def test_equality4():
    op1 = ParticleOperator("b10^ d1") + ParticleOperator("a0^")
    op2 = ParticleOperator("a0^") + ParticleOperator("b10^ d1")
    return op1 == op2


def test_order_indices_1():
    op = ParticleOperator("b0^ b1 b0")
    assert op == op.order_indices()


def test_order_indices_2():
    op = ParticleOperator("a1^ a0^")
    assert op == op.order_indices()


def test_order_indices_3():
    op = ParticleOperator("a0^ a1^")
    assert ParticleOperator("a1^ a0^") == op.order_indices()


def test_order_indices_4():
    op = (
        ParticleOperator("b1^ b0^ d2 a1")
        + ParticleOperator("a0")
        + ParticleOperator("b0^ b1^ d2 a1")
    )
    assert ParticleOperator("a0") == op.order_indices()


def test_is_hermitian():
    op = ParticleOperator("b0^ a1")
    assert op.is_hermitian == False


def test_is_hermitian2():
    op = ParticleOperator("b0^ b0")
    assert op.is_hermitian


def test_is_hermitian3():
    op = ParticleOperator("b1^ b0^ b1 b0")
    assert op.is_hermitian


def test_group_1():
    op = ParticleOperator("a1^ a2^") + ParticleOperator("a2^ a1^")
    assert op.group()[0] == 2 * ParticleOperator("a2^ a1^")


def test_group_2():
    op = ParticleOperator("b1^ b2^") + ParticleOperator("b2^ b1^")
    assert op.group() == []


def test_group_3():
    op = ParticleOperator("a1^ a0^ a1 a0")
    assert op.group() == [op]


def test_from_openfermion():
    from openfermion import FermionOperator as fo

    of_operator = fo("2^ 1") - fo("3") + 3 * fo("2^ 1^ 0^")
    op_operator = (
        ParticleOperator("b2^ b1")
        - ParticleOperator("b3")
        + 3 * ParticleOperator("b2^ b1^ b0^")
    )
    assert op_operator == ParticleOperator.from_openfermion(of_operator)


@pytest.mark.parametrize("n_terms", np.random.random_integers(1, 20, size=10))
def test_contains_returns_true_when_operator_is_present(n_terms):
    operator = ParticleOperator.random()
    contained_operator = np.random.choice(operator.to_list())
    assert operator.contains(contained_operator)


def test_contains_returns_false_when_operator_is_not_present():
    operator = ParticleOperator("b0 b1^ d3 a2")
    noncontained_operator = ParticleOperator("b0")
    assert not operator.contains(noncontained_operator)


def test_group_works_with_close_coefficients():
    operator = 1 * ParticleOperator("b0 b1^ d3 a2")
    operator += ((1 + 1e-13)) * operator.dagger()
    assert len(operator.group()) == 1
    assert len(operator.group()[0].to_list()) == 2


def test_mode_ordering_1():
    unordered = ParticleOperator("b0^ b2 b2^ b0")
    ordered = ParticleOperator("b2 b2^ b0^ b0")
    assert ordered == unordered.mode_order()


def test_mode_ordering_2():
    unordered = ParticleOperator("b1 b0 b3 b0^ b1^")
    ordered = (-1) ** 4 * ParticleOperator("b3 b1 b1^ b0 b0^")
    assert ordered == unordered.mode_order()


def test_mode_ordering_3():

    unordered = 3.1 * ParticleOperator("b3^ b0^ b2 b1") + 3.1 * ParticleOperator(
        "b2^ b1^ b3 b0"
    )
    expected = 3.1 * ParticleOperator("b3^ b2 b1 b0^") + 3.1 * ParticleOperator(
        "b3 b2^ b1^ b0"
    )
    assert unordered.mode_order() == expected


@pytest.mark.parametrize("trial", range(10))
def test_modes(trial):
    len_of_term = np.random.randint(1, 3)
    n_terms = np.random.randint(1, 3)
    modes = np.random.randint(0, 5, (n_terms, len_of_term))

    op_dict = {}

    for row in modes:
        key = tuple((0, val, 0) for val in row)
        op_dict[key] = 1

    assert np.allclose(modes, ParticleOperator(op_dict).modes)

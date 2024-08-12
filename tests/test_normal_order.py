from openparticle import ParticleOperator
import pytest
import numpy as np


def test_singleton():
    op1 = ParticleOperator("a0")
    op2 = ParticleOperator("b0^")
    op3 = ParticleOperator("d3")

    normal_order_op1 = op1.normal_order()
    normal_order_op2 = op2.normal_order()
    normal_order_op3 = op3.normal_order()

    assert op1.op_dict == normal_order_op1.op_dict
    assert op2.op_dict == normal_order_op2.op_dict
    assert op3.op_dict == normal_order_op3.op_dict


def test_identity_empty():
    op1 = ParticleOperator("")
    op2 = ParticleOperator(" ")
    op3 = ParticleOperator("            ")

    normal_order_op1 = op1.normal_order()
    normal_order_op2 = op2.normal_order()
    normal_order_op3 = op3.normal_order()

    assert op1.op_dict == normal_order_op1.op_dict
    assert op2.op_dict == normal_order_op2.op_dict
    assert op3.op_dict == normal_order_op3.op_dict


def test_is_normal_ordered_op():
    op = ParticleOperator("b0^ b1^ b0 b1 d0^ d1^ d0 d1 a0^ a1^ a0 a1")
    normal_order_op = op.normal_order()

    assert op.op_dict == normal_order_op.op_dict


def test_simply_fermion_same_mode():
    # b0 b0^ --> 1 - b0^ b0
    op = ParticleOperator("b0 b0^")
    normal_order_op = op.normal_order()
    expected = ParticleOperator({"": 1, "b0^ b0": -1})

    assert expected.op_dict == normal_order_op.op_dict


def test_simply_antifermion_same_mode():
    # d0 d0^ --> 1 - d0^ d0
    op = ParticleOperator("d0 d0^")
    normal_order_op = op.normal_order()
    expected = ParticleOperator({"": 1, "d0^ d0": -1})

    assert expected.op_dict == normal_order_op.op_dict


def test_simply_boson_same_mode():
    # a0 a0^ --> 1 + a0^ a0
    op = ParticleOperator("a0 a0^")
    normal_order_op = op.normal_order()
    expected = ParticleOperator({"": 1, "a0^ a0": 1})

    assert expected.op_dict == normal_order_op.op_dict


def test_simply_fermion_diff_mode():
    # b0 b1^ --> - b1^ b0
    op = ParticleOperator("b0 b1^")
    normal_order_op = op.normal_order()
    expected = ParticleOperator({"b1^ b0": -1})

    assert expected.op_dict == normal_order_op.op_dict


def test_simply_antifermion_diff_mode():
    # d0 d1^ --> - d1^ d0
    op = ParticleOperator("d0 d1^")
    normal_order_op = op.normal_order()
    expected = ParticleOperator({"d1^ d0": -1})

    assert expected.op_dict == normal_order_op.op_dict


def test_simply_boson_diff_mode():
    # a0 a1^ --> a1^ a0
    op = ParticleOperator("a0 a1^")
    normal_order_op = op.normal_order()
    expected = ParticleOperator({"a1^ a0": 1})

    assert expected.op_dict == normal_order_op.op_dict


def test_trivial_swap_fermions():
    op = ParticleOperator("b1 b2 b3 b4 b5^ b6^ b7^ b8^ b9^")
    normal_order_op = op.normal_order()
    expected = ParticleOperator({"b5^ b6^ b7^ b8^ b9^ b1 b2 b3 b4": 1})

    assert expected.op_dict == normal_order_op.op_dict


def test_trivial_swap_antifermions():
    op = ParticleOperator("d1 d2 d3 d4 d5^ d6^ d7^ d8^ d9^")
    normal_order_op = op.normal_order()
    expected = ParticleOperator({"d5^ d6^ d7^ d8^ d9^ d1 d2 d3 d4": 1})

    assert expected.op_dict == normal_order_op.op_dict


def test_trivial_swap_bosons():
    op = ParticleOperator("a1 a2 a3 a4 a5^ a6^ a7^ a8^ a9^")
    normal_order_op = op.normal_order()
    expected = ParticleOperator({"a5^ a6^ a7^ a8^ a9^ a1 a2 a3 a4": 1})

    assert expected.op_dict == normal_order_op.op_dict


def test_a_couple_fermions_1():
    op = ParticleOperator("b1 b2 b3 b3^")
    normal_order_op = op.normal_order()
    expected = ParticleOperator({"b1 b2": 1, "b3^ b1 b2 b3": -1})

    assert expected.op_dict == normal_order_op.op_dict


def test_a_couple_fermions_2():
    op = ParticleOperator("b0 b0^ b1")
    normal_order_op = op.normal_order()
    expected = ParticleOperator({"b1": 1, "b0^ b0 b1": -1})

    assert expected.op_dict == normal_order_op.op_dict


def test_a_couple_annihilation_particles_1():
    op = ParticleOperator("a0 b0 d0 a2 b2 d2")
    normal_order_op = op.normal_order()
    expected = ParticleOperator({"b0 b2 d0 d2 a0 a2": -1})

    assert expected.op_dict == normal_order_op.op_dict


def test_a_couple_particles():
    op = ParticleOperator("a1 a2 a3^ d1 d2^ b1 b2^")
    normal_order_op = op.normal_order()
    expected = ParticleOperator({"b2^ b1 d2^ d1 a3^ a1 a2": 1})

    assert expected.op_dict == normal_order_op.op_dict


def test_sign_change_between_b_and_d_1():
    op = ParticleOperator("d1 b1")
    normal_order_op = op.normal_order()
    expected = ParticleOperator({"b1 d1": -1})

    assert expected.op_dict == normal_order_op.op_dict


def test_sign_change_between_b_and_d_2():
    op = ParticleOperator("d1 b1^")
    normal_order_op = op.normal_order()
    expected = ParticleOperator({"b1^ d1": -1})

    assert expected.op_dict == normal_order_op.op_dict


def test_sign_change_between_b_and_d_3():
    op = ParticleOperator("d1^ b1")
    normal_order_op = op.normal_order()
    expected = ParticleOperator({"b1 d1^": -1})

    assert expected.op_dict == normal_order_op.op_dict


def test_sign_change_between_b_and_d_4():
    op = ParticleOperator("d1 b2 d2 d3 b3^")
    normal_order_op = op.normal_order()
    expected = ParticleOperator({"b3^ b2 d1 d2 d3": -1})

    assert expected.op_dict == normal_order_op.op_dict


def test_sign_change_between_b_and_d_5():
    op = ParticleOperator("d1 d2^ b1 b2^")
    normal_order_op = op.normal_order()
    expected = ParticleOperator({"b2^ b1 d2^ d1": 1})

    assert expected.op_dict == normal_order_op.op_dict


def test_complex_fermions_1():
    op = ParticleOperator("b1 b1 b1^ b1^")
    normal_order_op = op.normal_order()
    expected = ParticleOperator({})

    assert expected.op_dict == normal_order_op.op_dict


def test_complex_fermions_2():
    op = ParticleOperator("b1 b2 b1^ b2^")
    normal_order_op = op.normal_order()
    expected = ParticleOperator({"": -1, "b1^ b1": 1, "b2^ b2": 1, "b1^ b2^ b1 b2": 1})

    assert expected.op_dict == normal_order_op.op_dict


def test_complex_fermions_3():
    op = ParticleOperator("b1 b2 b2^ b1^")
    normal_order_op = op.normal_order()
    expected = ParticleOperator(
        {"": 1, "b1^ b1": -1, "b2^ b2": -1, "b2^ b1^ b1 b2": 1}
    )

    assert expected.op_dict == normal_order_op.op_dict


def test_complex_fermions_4():
    op = ParticleOperator("b1 b1 b1 b1^ b1^ b1^")
    normal_order_op = op.normal_order()
    expected = ParticleOperator({})

    assert expected.op_dict == normal_order_op.op_dict


def test_complex_fermions_5():
    op = ParticleOperator("b1 b1 b1^ b1^ b1^")
    normal_order_op = op.normal_order()
    expected = ParticleOperator({})

    assert expected.op_dict == normal_order_op.op_dict


def test_complex_fermions_6():
    op = ParticleOperator("b1 b1 b1 b1^")
    normal_order_op = op.normal_order()
    expected = ParticleOperator({})

    assert expected.op_dict == normal_order_op.op_dict


def test_complex_fermions_7():
    op = ParticleOperator("b1 b1 b1 b1^ b1^ b1^ b1 b1^ b1 b1^")
    normal_order_op = op.normal_order()
    expected = ParticleOperator({})

    assert expected.op_dict == normal_order_op.op_dict


def test_complex_fermions_8():
    op = ParticleOperator("b1 b1^ b1 b1 b1 b1^ b1^ b1^ b1 b1^ b1 b1^")
    normal_order_op = op.normal_order()
    expected = ParticleOperator({})

    assert expected.op_dict == normal_order_op.op_dict


def test_complex_fermions_9():
    op = ParticleOperator("b1 b2 b3 b3^ b2^ b1^")
    normal_order_op = op.normal_order()
    expected = ParticleOperator(
        {
            "": 1,
            "b1^ b1": -1,
            "b2^ b2": -1,
            "b2^ b1^ b1 b2": 1,
            "b3^ b3": -1,
            "b3^ b1^ b1 b3": 1,
            "b3^ b2^ b2 b3": 1,
            "b3^ b2^ b1^ b1 b2 b3": -1,
        }
    )

    assert expected.op_dict == normal_order_op.op_dict


def test_complex_fermions_10():
    op = ParticleOperator("b1 b1^ b1 b1^ b1 b1^ b1 b1^")
    normal_order_op = op.normal_order()
    expected = ParticleOperator(
        {
            "": 1,
            "b1^ b1": -1
        }
    )

    assert expected.op_dict == normal_order_op.op_dict


def test_complex_bosons():
    op = ParticleOperator("a1 a1 a1^ a1^")
    normal_order_op = op.normal_order()
    expected = ParticleOperator({"": 2, "a1^ a1": 4, "a1^ a1^ a1 a1": 1})

    assert expected.op_dict == normal_order_op.op_dict


def test_normal_order_sum_1():
    op = ParticleOperator("b0 b0^ d1") + ParticleOperator("d1")
    expected = ParticleOperator({"b0^ b0 d1": -1, "d1": 2})
    assert op.normal_order().op_dict == expected.op_dict


def test_normal_order_sum_2():
    op = ParticleOperator("b0 b0^ d1") - ParticleOperator("d1")
    expected = ParticleOperator({"b0^ b0 d1": -1})
    assert op.normal_order().op_dict == expected.op_dict


def test_normal_order_sum_3():
    op = ParticleOperator("a0 a1^") - ParticleOperator("a1^ a0")
    assert op.normal_order().op_dict == {}


def test_commutation_relations_bosonic():
    # [a_i, a_j^\dagger] = \delta_ij
    op = ParticleOperator("a0").commutator(ParticleOperator("a0^"))
    identity = ParticleOperator("")
    assert op.op_dict == identity.op_dict


def test_commutation_relations_bosonic_2():
    op = ParticleOperator("a0").commutator(ParticleOperator("a1^"))
    identity = ParticleOperator({})
    assert op.op_dict == identity.op_dict


def test_commutation_relations_fermionic_1():
    # {b_i, b_j^\dagger} = \delta_ij
    op = ParticleOperator("b0").anticommutator(ParticleOperator("b0^"))
    identity = ParticleOperator("")
    assert op.op_dict == identity.op_dict


def test_commutation_relations_fermionic_2():
    # {b_i, d_j^\dagger} = 0
    op = ParticleOperator("b0").anticommutator(ParticleOperator("d0^"))
    comm = ParticleOperator({})
    assert op.op_dict == comm.op_dict


def test_mixed_commutator():
    op = ParticleOperator("b0").commutator(ParticleOperator("a0^"))
    comm = ParticleOperator({})
    assert op.op_dict == comm.op_dict


def test_commutation_relations_1():
    op = ParticleOperator("b0^ b0 a0^").commutator(ParticleOperator("b0^ b0"))
    comm = ParticleOperator({})
    assert op.op_dict == comm.op_dict


def test_commutation_relations_2():
    op = ParticleOperator("b0^ b0 a0^").commutator(ParticleOperator("a0^ a0"))
    comm = ParticleOperator({"b0^ b0 a0^": -1})
    assert op.op_dict == comm.op_dict


def test_commutation_relations_3():
    op = ParticleOperator("b0^ b0 a0^").commutator(ParticleOperator("b0^ b0 a0"))
    comm = ParticleOperator({"b0^ b0": -1})
    assert op.op_dict == comm.op_dict


def test_commutation_relations_4():
    op = ParticleOperator("b0^ b0 a0").commutator(ParticleOperator("b0^ b0"))
    comm = ParticleOperator({})
    assert op.op_dict == comm.op_dict


def test_commutation_relations_5():
    op = ParticleOperator("b0^ b0 a0").commutator(ParticleOperator("b0^ b0 a0^"))
    comm = ParticleOperator("b0^ b0")
    assert op.op_dict == comm.op_dict


def test_commutation_relations_6():
    op = ParticleOperator("b0^ b0 a0").commutator(ParticleOperator("a0^ a0"))
    comm = ParticleOperator("b0^ b0 a0")
    assert op.op_dict == comm.op_dict


def test_commutation_relations():
    op1 = ParticleOperator("b0^ b0 a0^") - ParticleOperator("b0^ b0 a0")
    op2 = (
        ParticleOperator("b0^ b0")
        + ParticleOperator("a0^ a0")
        + ParticleOperator("b0^ b0 a0^")
        + ParticleOperator("b0^ b0 a0")
    )
    comm = (
        -1 * ParticleOperator("b0^ b0 a0^")
        - ParticleOperator("b0^ b0 a0")
        - 2 * ParticleOperator("b0^ b0")
    )
    assert op1.commutator(op2).op_dict == comm.op_dict

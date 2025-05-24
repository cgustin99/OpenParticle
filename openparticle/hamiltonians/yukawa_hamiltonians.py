import numpy as np
from openparticle.fields import ScalarField, FermionField
from openparticle.dlcq import *
from openparticle import ParticleOperator
from openparticle.utils import remove_symmetry_terms
from openparticle.hamiltonians.free_hamiltonians import (
    free_boson_Hamiltonian,
    free_fermion_Hamiltonian,
)
from itertools import product
import time


def index_to_k(operator):
    # Returns momentum variable k for given operator and index
    # e.g. b_0 returns k = 1/2, a2 returns k = 3
    if operator.has_fermions or operator.has_antifermions:
        return operator.modes[0][0] + 1 / 2
    elif operator.has_bosons:
        return operator.modes[0][0] + 1


def resolution_constraint(terms, resolution):
    # Remove terms with sum of all k created or annihilated > K
    good_terms = {}

    for term in terms:
        k_created, k_annihilated = 0, 0
        for op in term.split():
            k = index_to_k(op)
            if op.creation:
                k_created += k
            else:
                k_annihilated += k
        if k_created <= resolution and k_annihilated <= resolution:
            key, val = list(term.op_dict.keys())[0], list(term.op_dict.values())[0]
            good_terms[key] = val

    return ParticleOperator(good_terms)


# Yukawa Interaction
def three_point_yukawa(res, g, mf, mb):

    fermion_lim = res - ((res - 0.5) % 1)
    boson_lim = int(res)
    fermionic_range = np.arange(-fermion_lim, fermion_lim + 1, 1)
    bosonic_range = np.array([i for i in range(-boson_lim, boson_lim + 1) if i != 0])

    L = 2 * np.pi * res
    container_dict = dict()
    for k1, k2, k3 in product(fermionic_range, fermionic_range, bosonic_range):
        if k1 == k2 + k3:
            op = (
                FermionField(k1, L, mf).psi_dagger.dot(
                    gamma0.dot(FermionField(k2, L, mf).psi)
                )[0][0]
                * ScalarField(k3, L, mb).phi
            )
            if len(op.op_dict) != 0:
                helper_variable = op.normal_order()
                for op_str, coeff in helper_variable.op_dict.items():
                    container_dict[op_str] = coeff + container_dict.get(op_str, 0.0)
    three_point = ParticleOperator(container_dict)
    three_point.remove_identity()
    three_point = resolution_constraint(three_point, res)
    return 2 * L * g / (2 * L) ** 3 * three_point  # * (P+ = 1)


def instantaneous_yukawa(res, g, mf, mb):

    fermion_lim = res - ((res - 0.5) % 1)
    boson_lim = int(res)
    fermionic_range = np.arange(-fermion_lim, fermion_lim + 1, 1)
    bosonic_range = np.array([i for i in range(-boson_lim, boson_lim + 1) if i != 0])

    L = 2 * np.pi * res
    container_dict = dict()

    for k1, k2, k3, k4 in product(
        fermionic_range, bosonic_range, bosonic_range, fermionic_range
    ):
        if k1 == k2 + k3 + k4:
            left = FermionField(k1, L, mf).psi_dagger * ScalarField(k2, L, mb).phi
            right = FermionField(k4, L, mf).psi * ScalarField(k3, L, mb).phi
            op = 1 / (p(k3, L) + p(k4, L)) * (left.dot(Lambdap.dot(right))[0][0])
            if len(op.op_dict) != 0:
                helper_variable = op.normal_order()
                for op_str, coeff in helper_variable.op_dict.items():
                    container_dict[op_str] = coeff + container_dict.get(op_str, 0.0)
    four_point = ParticleOperator(container_dict)

    four_point.remove_identity()
    four_point = remove_symmetry_terms(four_point, 4)
    four_point = resolution_constraint(four_point, res)

    return 2 * L * g**2 / (2 * L) ** 4 * four_point  # * (P+ = 1)


def instantaneous_yukawa_dummy(res):

    fermion_lim = res - ((res - 0.5) % 1)
    boson_lim = int(res)
    fermionic_range = np.arange(-fermion_lim, fermion_lim + 1, 1)
    bosonic_range = np.array([i for i in range(-boson_lim, boson_lim + 1) if i != 0])

    L = 2 * np.pi * res
    container_dict = dict()

    for k1, k2, k3, k4 in product(
        fermionic_range, bosonic_range, bosonic_range, fermionic_range
    ):
        if k1 == k2 + k3 + k4:
            op1 = (k1 > 0) * ParticleOperator("b" + str(abs(int(k1 - 0.5))) + "^") + (
                k1 < 0
            ) * ParticleOperator("d" + str(abs(int(k1 - 0.5))))
            op2 = (k2 > 0) * ParticleOperator("a" + str(abs(int(k2 - 1)))) + (
                k2 < 0
            ) * ParticleOperator("a" + str(abs(int(k2 + 1))) + "^")
            op3 = (k3 > 0) * ParticleOperator("a" + str(abs(int(k3 - 1)))) + (
                k3 < 0
            ) * ParticleOperator("a" + str(abs(int(k3 + 1))) + "^")
            op4 = (k4 > 0) * ParticleOperator("b" + str(abs(int(k4 - 0.5)))) + (
                k4 < 0
            ) * ParticleOperator("d" + str(abs(int(k4 - 0.5))) + "^")
            op = (op1 * op2 * op3 * op4).normal_order().order_indices()
            op = remove_symmetry_terms(op, 4)
            op = (1 / op.coeff) * op
            if len(op.op_dict) != 0:
                helper_variable = op
                for op_str, coeff in helper_variable.op_dict.items():
                    container_dict[op_str] = coeff + container_dict.get(op_str, 0.0)
    four_point = ParticleOperator(container_dict)

    four_point.remove_identity()
    four_point = remove_symmetry_terms(four_point, 4)
    four_point = resolution_constraint(four_point, res)

    return four_point


def yukawa_hamiltonian(res, g, mf, mb):
    Ham = (
        free_boson_Hamiltonian(res=res, mb=mb)
        + free_fermion_Hamiltonian(res=res, mf=mf)
        + three_point_yukawa(res=res, g=g, mf=mf, mb=mb)
        + instantaneous_yukawa(res=res, g=g, mf=mf, mb=mb)
    )
    return Ham

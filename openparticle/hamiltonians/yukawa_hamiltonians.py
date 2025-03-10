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

    return 2 * L * g**2 / (2 * L) ** 4 * four_point  # * (P+ = 1)


def yukawa_hamiltonian(res, g, mf, mb):
    Ham = (
        free_boson_Hamiltonian(res=res, mb=mb)
        + free_fermion_Hamiltonian(res=res, mf=mf)
        + three_point_yukawa(res=res, g=g, mf=mf, mb=mb)
        + instantaneous_yukawa(res=res, g=g, mf=mf, mb=mb)
    )
    return Ham

import numpy as np
from openparticle.fields import ScalarField, FermionField
from openparticle.dlcq import *
from openparticle import ParticleOperator
from openparticle.utils import remove_symmetry_terms
from openparticle.hamiltonians.free_hamiltonians import (
    free_boson_Hamiltonian,
    free_fermion_Hamiltonian,
)


# Yukawa Interaction
def three_point_yukawa(res, g, mf, mb):

    L = 2 * np.pi * res
    three_point = ParticleOperator({})
    for k1 in np.arange(-res + 1 / 2, res + 1 / 2, 1):
        for k2 in np.arange(-res + 1 / 2, res + 1 / 2, 1):
            for k3 in [i for i in range(-res, res + 1) if i != 0]:
                if k1 == k2 + k3:
                    op = (
                        FermionField(k1, L, mf).psi_dagger.dot(
                            gamma0.dot(FermionField(k2, L, mf).psi)
                        )[0][0]
                        * ScalarField(k3, L, mb).phi
                    )
                    if len(op.op_dict) != 0:
                        three_point += op.normal_order()

    three_point.remove_identity()
    return 2 * L * g / (2 * L) ** 3 * three_point


def instantaneous_yukawa(res, g, mf, mb):
    L = 2 * np.pi * res
    four_point = ParticleOperator({})

    for k1 in np.arange(-res + 1 / 2, res + 1 / 2, 1):
        for k2 in [i for i in range(-res, res + 1) if i != 0]:
            for k3 in [i for i in range(-res, res + 1) if i != 0]:
                for k4 in np.arange(-res + 1 / 2, res + 1 / 2, 1):
                    if k1 == k2 + k3 + k4:
                        left = (
                            FermionField(k1, L, mf).psi_dagger
                            * ScalarField(k2, L, mb).phi
                        )
                        right = FermionField(k4, L, mf).psi * ScalarField(k3, L, mb).phi
                        op = (
                            1
                            / (p(k3, L) + p(k4, L))
                            * (left.dot(Lambdap.dot(right))[0][0])
                        )
                        if len(op.op_dict) != 0:
                            four_point += op.normal_order()

    four_point.remove_identity()
    four_point = remove_symmetry_terms(four_point, 4)

    return 0.5 * 2 * L * g**2 / (2 * L) ** 4 * four_point


def yukawa_Hamiltonian(res, g, mf, mb):
    Ham = (
        free_boson_Hamiltonian(res=res, mb=mb)
        + free_fermion_Hamiltonian(res=res, mf=mf)
        + three_point_yukawa(res=res, g=g, mf=mf, mb=mb)
        + instantaneous_yukawa(res=res, g=g, mf=mf, mb=mb)
    )
    return Ham

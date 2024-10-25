import numpy as np
from openparticle.fields import ScalarField, FermionField
from openparticle.dlcq import *
from openparticle import ParticleOperator
from openparticle.utils import remove_symmetry_terms


# Non-interacting Hamiltonians
def free_boson_Hamiltonian(Lambda, m_B=1, L=2 * np.pi):
    free_scalar = ParticleOperator({})
    for k in [i for i in range(-Lambda, Lambda + 1) if i != 0]:
        free_scalar += (
            ScalarField(-k, m_B, L).phi * ScalarField(k, m_B, L).phi
        ).normal_order()
    free_scalar.remove_identity()

    return 2 * np.pi * m_B**2 / (4 * L**2) * free_scalar


def free_fermion_Hamiltonian(Lambda, m_F=1, L=1):
    free_fermi = np.array([ParticleOperator({})]).reshape([1, -1])
    for k in np.arange(-Lambda + 1 / 2, Lambda + 1 / 2):
        free_fermi += (
            1
            / (p(k, L=L))
            * (
                FermionField(k, m_F, L).psi_dagger.dot(
                    Lambdap.dot(FermionField(k, m_F, L).psi)
                )
            )[0][0].normal_order()
        )

    free_fermi[0][0].remove_identity()

    return 2 * m_F**2 / (4 * L**2) * free_fermi[0][0]


# Yukawa Interaction
def three_point_yukawa(Lambda, g=1, L=1):
    three_point = ParticleOperator({})
    for k1 in np.arange(-Lambda + 1 / 2, Lambda + 1 / 2):
        for k2 in np.arange(-Lambda + 1 / 2, Lambda + 1 / 2):
            for k3 in [i for i in range(-Lambda, Lambda + 1) if i != 0]:
                if k1 == k2 + k3:
                    op = (
                        FermionField(k1).psi_dagger.dot(
                            gamma0.dot(FermionField(k2).psi)
                        )[0][0]
                        * ScalarField(k3).phi
                    )
                    if len(op.op_dict) != 0:
                        three_point += op.normal_order()

    three_point.remove_identity()
    return 4 * np.pi * g / (2 * L) ** 3 * three_point


def instantaneous_yukawa(Lambda, g=1, L=1):
    four_point = ParticleOperator({})

    for k1 in np.arange(-Lambda + 1 / 2, Lambda + 1 / 2, 1):
        for k2 in [i for i in range(-Lambda, Lambda + 1) if i != 0]:
            for k3 in [i for i in range(-Lambda, Lambda + 1) if i != 0]:
                for k4 in np.arange(-Lambda + 1 / 2, Lambda + 1 / 2, 1):
                    if k1 == k2 + k3 + k4:
                        left = FermionField(k1).psi_dagger * ScalarField(k2).phi
                        right = FermionField(k4).psi * ScalarField(k3).phi
                        op = 1 / (p(k3) + p(k4)) * (left.dot(Lambdap.dot(right))[0][0])
                        if len(op.op_dict) != 0:
                            four_point += op.normal_order()

    four_point.remove_identity()
    four_point = remove_symmetry_terms(four_point, 4)

    # return g**2 / (2 * L) ** 3 * four_point
    return 4 * np.pi * g**2 / (2 * L) ** 4 * four_point


def yukawa_Hamiltonian(Lambda, g=1, L=1, m_B=1, m_F=1):
    Ham = (
        free_boson_Hamiltonian(Lambda=Lambda, m_B=m_B, L=L)
        + free_fermion_Hamiltonian(Lambda=Lambda, m_F=m_F, L=L)
        + three_point_yukawa(Lambda=Lambda, g=g, L=L)
        + instantaneous_yukawa(Lambda=Lambda, g=g, L=L)
    )
    return Ham

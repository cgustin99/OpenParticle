import numpy as np
from openparticle.fields import ScalarField, FermionField
from openparticle.dlcq import *
from openparticle import ParticleOperator


# Non-interacting Hamiltonians
def free_boson_Hamiltonian(Lambda, m_B=1, L=1):
    free_scalar = ParticleOperator({})
    for k in [i for i in range(-Lambda, Lambda + 1) if i != 0]:
        free_scalar += (
            ScalarField(-k, m_B, L).phi * ScalarField(k, m_B, L).phi
        ).normal_order()
    free_scalar.remove_identity()

    return L * m_B**2 / (2 * np.pi) * free_scalar


def free_fermion_Hamiltonian(Lambda, m_F=1, L=1):
    free_fermi = np.array([ParticleOperator({})]).reshape([1, -1])
    for k in np.arange(-Lambda + 1 / 2, Lambda):
        free_fermi += (
            m_F**2
            / (2 * L * p(k))
            * (
                FermionField(k, m_F, L).psi_dagger.dot(
                    Lambdap.dot(FermionField(k, m_F, L).psi)
                )
            )[0][0].normal_order()
        )

    free_fermi[0][0].remove_identity()

    return L * m_F**2 / (2 * np.pi) * free_fermi[0][0]


# Yukawa Interaction
def three_point_yukawa(Lambda, g=1, L=1):
    three_point = ParticleOperator({})
    for k1 in np.arange(-Lambda + 1 / 2, Lambda):
        for k2 in np.arange(-Lambda + 1 / 2, Lambda):
            for k3 in [i for i in range(-Lambda, Lambda + 1) if i != 0]:
                if k1 == k2 + k3:
                    three_point += (
                        FermionField(k1).psi_dagger.dot(
                            gamma0.dot(FermionField(k2).psi)
                        )[0][0]
                        * ScalarField(k3).phi
                    ).normal_order()

    three_point.remove_identity()
    return g / (2 * L) ** 2 * three_point


def instantaneous_yukawa(Lambda, g=1, L=1):
    four_point = ParticleOperator({})

    for k1 in np.arange(-Lambda + 1 / 2, Lambda):
        for k2 in [i for i in range(-Lambda, Lambda + 1) if i != 0]:
            for k3 in [i for i in range(-Lambda, Lambda + 1) if i != 0]:
                for k4 in np.arange(-Lambda + 1 / 2, Lambda, 1):
                    if k1 == k2 + k3 + k4:
                        four_point += (
                            1
                            / (p(k3) + p(k4))
                            * (
                                (FermionField(k1).psi_dagger * ScalarField(k2).phi).dot(
                                    Lambdap.dot(
                                        ScalarField(k3).phi * FermionField(k4).psi
                                    )
                                )
                            )
                        )[0][0].normal_order()
    four_point.remove_identity()

    return g**2 / (2 * L) ** 3 * four_point


def yukawa_Hamiltonian(Lambda, g=1, L=1, m_B=1, m_F=1):
    Ham = (
        free_boson_Hamiltonian(Lambda=Lambda, m_B=m_B, L=L)
        + free_fermion_Hamiltonian(Lambda=Lambda, m_F=m_F, L=L)
        + three_point_yukawa(Lambda=Lambda, g=g, L=L)
        + instantaneous_yukawa(Lambda=Lambda, g=g, L=L)
    )
    return Ham

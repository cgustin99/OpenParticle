import numpy as np
from openparticle.fields import ScalarField, FermionField
from openparticle.dlcq import *
from openparticle import ParticleOperator
from openparticle.utils import remove_symmetry_terms


# Non-interacting Hamiltonians
def free_boson_Hamiltonian(res, mb):

    L = 2 * np.pi * res  # P+ = 1

    H_free_scalar = ParticleOperator({})
    lim = int(res)
    for k in [i for i in range(-lim, lim + 1) if i != 0]:
        H_free_scalar += (
            ScalarField(-k, L, mb).phi * ScalarField(k, L, mb).phi
        ).normal_order()
    H_free_scalar.remove_identity()
    return L * mb**2 / (2 * L) ** 2 * H_free_scalar


def free_fermion_Hamiltonian(res, mf):

    L = 2 * np.pi * res  # P+ = 1

    H_free_fermion = ParticleOperator({})
    lim = res - ((res - 0.5) % 1)
    for k in np.arange(-lim, lim + 1, 1):
        H_free_fermion += (
            1
            / p(k, L)
            * (
                FermionField(k, L, mf).psi_dagger.dot(
                    Lambdap.dot(FermionField(k, L, mf).psi)
                )
            )[0][0].normal_order()
        )
    H_free_fermion.remove_identity()
    return 2 * L * mf**2 / (2 * L) ** 2 * H_free_fermion

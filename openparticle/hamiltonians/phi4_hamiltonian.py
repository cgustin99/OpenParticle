import numpy as np
from openparticle.fields import ScalarField
from openparticle.hamiltonians.free_hamiltonians import free_boson_Hamiltonian
from openparticle.dlcq import *
from openparticle import ParticleOperator
from openparticle.utils import remove_symmetry_terms


def phi4_interaction(res, g, mb):
    L = 2 * np.pi * res
    phi4 = ParticleOperator({})

    for k1 in [i for i in range(-res, res + 1) if i != 0]:
        for k2 in [i for i in range(-res, res + 1) if i != 0]:
            for k3 in [i for i in range(-res, res + 1) if i != 0]:
                for k4 in [i for i in range(-res, res + 1) if i != 0]:
                    if k1 + k2 + k3 + k4 == 0:
                        op = (
                            ScalarField(k1, L, mb).phi
                            * ScalarField(k2, L, mb).phi
                            * ScalarField(k3, L, mb).phi
                            * ScalarField(k4, L, mb).phi
                        )

                        if len(op.op_dict) != 0:
                            phi4 += op.normal_order()

    phi4.remove_identity()
    return 2 * L * g / (2 * L) ** 4 * phi4


def phi4_Hamiltonian(res, g, mb):
    Ham = free_boson_Hamiltonian(res=res, mb=mb) + phi4_interaction(res=res, g=g, mb=mb)
    return Ham

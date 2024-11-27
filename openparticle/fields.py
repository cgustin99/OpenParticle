from openparticle import ParticleOperator
import numpy as np
from openparticle.dlcq import *


class ScalarField:

    def __init__(self, k, L, mb):
        self.k = k
        self.mb = mb
        self.L = L
        self.phi = (
            (2 * L)
            / np.sqrt(4 * np.pi * np.abs(k))
            * (
                np.heaviside(k, 0) * ParticleOperator("a" + str(k - 1))
                + np.heaviside(-k, 0) * ParticleOperator("a" + str(-k - 1) + "^")
            )
        )


class FermionField:

    def __init__(self, k, L, mf):
        self.k = k
        self.mf = mf
        self.L = L
        self.psi = (
            (2 * L)
            / np.sqrt(4 * np.pi * np.abs(k))
            * (
                np.heaviside(k, 0)
                * u(p(k, L), mf)
                * ParticleOperator("b" + str(int(k - 1 / 2)))
                + np.heaviside(-k, 0)
                * v(p(-k, L), mf)
                * ParticleOperator("d" + str(-int(k + 1 / 2)) + "^")
            )
        )

        self.psi_dagger = (
            (2 * L)
            / np.sqrt(4 * np.pi * np.abs(k))
            * (
                np.heaviside(k, 0)
                * (u(p(k, L), mf).reshape([1, -1]))
                * ParticleOperator("b" + str(int(k - 1 / 2)) + "^")
                + np.heaviside(-k, 0)
                * (v(p(-k, L), mf).reshape([1, -1]))
                * ParticleOperator("d" + str(-int(k + 1 / 2)))
            )
        )

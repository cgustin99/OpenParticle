from openparticle import ParticleOperator
import numpy as np
from openparticle.dlcq import *


class ScalarField:

    def __init__(self, k, L, m_B=1):
        self.k = k
        self.m_B = m_B
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

    def __init__(self, k, L, m_F=1):
        self.k = k
        self.m_F = m_F
        self.L = L
        self.psi = (
            (2 * L)
            / np.sqrt(4 * np.pi * np.abs(k))
            * (
                np.heaviside(k, 0)
                * u(p(k, L), m_F)
                * ParticleOperator("b" + str(int(k - 1 / 2)))
                + np.heaviside(-k, 0)
                * v(p(-k, L), m_F)
                * ParticleOperator("d" + str(-int(k + 1 / 2)) + "^")
            )
        )

        self.psi_dagger = (
            (2 * L)
            / np.sqrt(4 * np.pi * np.abs(k))
            * (
                np.heaviside(k, 0)
                * (u(p(k, L), m_F).reshape([1, -1]))
                * ParticleOperator("b" + str(int(k - 1 / 2)) + "^")
                + np.heaviside(-k, 0)
                * (v(p(-k, L), m_F).reshape([1, -1]))
                * ParticleOperator("d" + str(-int(k + 1 / 2)))
            )
        )

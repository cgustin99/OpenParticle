from openparticle import Fock
import numpy as np
from typing import List


def get_computational_states_fermionic(n_modes):
    return [
        op.Fock([int(j) for j in format(i, "0" + str(n_modes) + "b")][::-1], [], [])
        for i in range(2**n_modes)
    ]


def generate_matrix_from_basis(operator: op.ParticleOperator, basis: List[Fock]):
    size = (len(basis), len(basis))
    matrix = np.zeros(size)

    for i, state_i in enumerate(basis):
        for j, state_j in enumerate(basis):
            matrix_element = state_i.dagger() * operator * state_j
            matrix[i][j] = matrix_element

    return matrix

import openparticle as op
import numpy as np
from typing import List


def get_computational_states_fermionic(n_modes):
    return [
        op.Fock([int(j) for j in format(i, "0" + str(n_modes) + "b")][::-1], [], [])
        for i in range(2**n_modes)
    ]


def generate_matrix_from_basis(operator: op.ParticleOperator, basis: List[op.Fock]):
    size = (len(basis), len(basis))
    matrix = np.zeros(size)

    for i, state_i in enumerate(basis):
        for j, state_j in enumerate(basis):
            matrix_element = state_i.dagger() * operator * state_j
            matrix[i][j] = matrix_element

    return matrix


def latexify(string):
    # Split terms on whitespace
    terms = string.split()

    processed_terms = []
    for term in terms:
        # Insert underscores between letters and numbers
        term = re.sub(r"([a-zA-Z])(\d)", r"\1_\2", term)
        term = re.sub(r"(\d)([a-zA-Z])", r"\1_\2", term)

        # Add † symbol after any term with ^
        if "^" in term:
            term += "†"

        processed_terms.append(term)

    return " ".join(processed_terms)

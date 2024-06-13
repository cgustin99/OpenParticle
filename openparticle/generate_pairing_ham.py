import openparticle as op
import numpy as np


def generate_fermioinic_pairing_hamiltonian(N):
    H = op.ParticleOperator('b0^ b0')

    for i in range(N + 1):
        for j in range(N + 1):
            if i == j == 0:
                coeff = 0
            else: coeff = 1

            H += coeff * op.ParticleOperator('b' + str(i) + '^ ' + 'b' + str(j))

    return H

def generate_all_combos_of_fermionic_fock_states(N_modes):
    N = N_modes + 1
    all_states = []
    for i in range(2**N):
        binary_string = [int(j) for j in format(i, f'0{N}b')]
        f_occ = list(np.nonzero(binary_string)[0])
        all_states.append(op.Fock(f_occ, [], []))

    return all_states

def generate_matrix_from_basis(operator, basis):

    size = (len(basis), len(basis))
    matrix = np.zeros(size)

    for i, state_i in enumerate(basis):
        for j, state_j in enumerate(basis):
            matrix[i][j] = state_i.dagger() * operator * state_j

    return matrix

def generate_pairing_hamiltonian(max_modes):
    states = generate_all_combos_of_fermionic_fock_states(max_modes)
    h_op = generate_fermioinic_pairing_hamiltonian(max_modes)
    return generate_matrix_from_basis(h_op, states)
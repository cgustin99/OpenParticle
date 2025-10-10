import numpy as np
from openparticle import ParticleOperator
from openparticle.utils import get_fock_basis, generate_matrix
from scipy.linalg import expm


def displacement_operator(alpha, Lambda):
    a = generate_matrix(
        ParticleOperator("a0"), get_fock_basis(ParticleOperator("a0"), Lambda)
    )
    adagger = generate_matrix(
        ParticleOperator("a0^"), get_fock_basis(ParticleOperator("a0^"), Lambda)
    )

    return expm(alpha * adagger - alpha.conjugate() * a)


def get_boson_state(n_bosons, cutoff):
    boson_state = np.zeros(cutoff + 1).reshape(-1, 1)
    boson_state[n_bosons] = 1
    return boson_state


def get_fermion_state(n_fermions):
    if n_fermions == 0:
        return np.array([[1, 0]]).reshape(-1, 1)
    elif n_fermions == 1:
        return np.array([[0, 1]]).reshape(-1, 1)


def get_initial_state(fermion_state, boson_state, cutoff, g):
    if fermion_state == "single_fermion":
        fermion = get_fermion_state(1)
    elif fermion_state == "plus_fermion":
        fermion = get_fermion_state(0) + get_fermion_state(1)
        fermion = 1 / np.linalg.norm(fermion) * fermion

    if boson_state == "zero_boson":
        boson = get_boson_state(0, cutoff)
    elif boson_state == "coherent_boson":
        boson = displacement_operator(
            alpha=-1j * g / 2, Lambda=cutoff
        ) @ get_boson_state(0, cutoff)

    return np.kron(fermion, boson)

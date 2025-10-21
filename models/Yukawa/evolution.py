from openparticle import ParticleOperator
from openparticle.utils import get_fock_basis, generate_matrix
from scipy.linalg import expm
import numpy as np
from symmer import PauliwordOp
from functools import reduce


def naive_interaction_gate(g, t, Lambda):
    a_dagger_plus_a_matrix = generate_matrix(
        ParticleOperator("a0^") + ParticleOperator("a0"),
        get_fock_basis(ParticleOperator("a0^") + ParticleOperator("a0"), Lambda),
    )
    Z = PauliwordOp.from_dictionary({"Z": 1.0}).to_sparse_matrix.toarray()
    I = PauliwordOp.from_dictionary({"I": 1.0}).to_sparse_matrix.toarray()
    b_dagger_b_matrix = (I - Z) / 2

    interaction_term = np.kron(b_dagger_b_matrix, a_dagger_plus_a_matrix)
    return expm(-1j * g * t * interaction_term)


def padded_fermion_Rz_gate(Eb, t, Lambda):
    # exp(-1j theta b^dagger b \otimes I...)
    Z = PauliwordOp.from_dictionary({"Z": 1.0}).to_sparse_matrix.toarray()
    I = PauliwordOp.from_dictionary({"I": 1.0}).to_sparse_matrix.toarray()
    b_dagger_b = (I - Z) / 2
    b_dagger_b_padded = reduce(np.kron, [b_dagger_b, np.eye(Lambda + 1)])

    return expm(-1j * Eb * t * b_dagger_b_padded)


def padded_phase_space_rotation_gate(Ef, t, Lambda):
    # exp(-1j theta I.. \otimes a^dagger a)
    a_dagger_a_matrix = generate_matrix(
        ParticleOperator("a0^ a0"), get_fock_basis(ParticleOperator("a0^ a0"), Lambda)
    )
    padded_a_dagger_a_matrix = np.kron(np.eye(2), a_dagger_a_matrix)
    return expm(-1j * Ef * t * padded_a_dagger_a_matrix)


def exact_unitary(Ef, Eb, g, t, Lambda):
    ham_op = (
        Ef * ParticleOperator("b0^ b0")
        + Eb * ParticleOperator("a0^ a0")
        + g
        * ParticleOperator("b0^ b0")
        * (ParticleOperator("a0^") + ParticleOperator("a0"))
    )
    ham_op_matrix = generate_matrix(ham_op, get_fock_basis(ham_op, Lambda), Lambda)
    return expm(-1j * t * ham_op_matrix)


def first_order_trotter(Ef, Eb, g, t, Lambda):
    R_theta_boson_gate = padded_phase_space_rotation_gate(Ef, t, Lambda)
    R_theta_fermion_gate = padded_fermion_Rz_gate(Eb, t, Lambda)
    R_theta_interaction_gate = naive_interaction_gate(g, t, Lambda)

    return R_theta_boson_gate @ R_theta_fermion_gate @ R_theta_interaction_gate


def second_order_trotter(Ef, Eb, g, t, Lambda):
    trotter_unitary = padded_phase_space_rotation_gate(Ef, t / 2, Lambda)
    trotter_unitary = padded_fermion_Rz_gate(Eb, t / 2, Lambda) @ trotter_unitary
    trotter_unitary = naive_interaction_gate(g, t, Lambda) @ trotter_unitary
    trotter_unitary = padded_fermion_Rz_gate(Eb, t / 2, Lambda) @ trotter_unitary
    trotter_unitary = (
        padded_phase_space_rotation_gate(g, t / 2, Lambda) @ trotter_unitary
    )

    return trotter_unitary


def _higher_order_trotter_coeff(order):
    return (4 - 4 ** (1 / (order - 1))) ** (-1)


def trotterized_unitary(Ef, Eb, g, t, Lambda, order=1, n_steps=1):
    if n_steps > 1:
        unitary = trotterized_unitary(
            Ef, Eb, g, t / n_steps, Lambda, order=order, n_steps=1
        )
        return np.linalg.matrix_power(unitary, n_steps)
    if order == 1:
        return first_order_trotter(Ef, Eb, g, t, Lambda)
    elif order == 2:
        return second_order_trotter(Ef, Eb, g, t, Lambda)
    elif order == 0:
        return exact_unitary(Ef, Eb, g, t, Lambda)
    else:
        assert order % 2 == 0
        coeff = _higher_order_trotter_coeff(order)
        left_op = trotterized_unitary(Ef, Eb, g, coeff * t, Lambda, order - 2)
        middle_op = trotterized_unitary(
            Ef, Eb, g, (1 - 4 * coeff) * t, Lambda, order - 2
        )
        return left_op @ left_op @ middle_op @ left_op @ left_op


def evolve_statevector(t, Ef, Eb, g, Lambda, order, n_steps, statevector):
    return (
        trotterized_unitary(
            Ef=Ef, Eb=Eb, g=g, t=t, Lambda=Lambda, order=order, n_steps=n_steps
        )
        @ statevector
    )


def estimate_cost(order, n_steps):
    """Estimate the resources required to implement the Trotter approximation.

    Args:
        order (int): order of the Suzuki-Trotter decomposition
        n_steps (int): number of Trotter steps (Lie-Trotter)

    Returns:
        Tuple:
            int: number of queries to the first-order decomposition (does not account for
                merging of neighboring commuting evolutions)
            int: number of sdf gates
    """
    num_queries_per_step = 0
    num_sdf_gates = 0
    if order > 2:
        num_queries_per_step = 2 * (5 ** ((order / 2) - 1))
        num_sdf_gates = 5 ** ((order / 2) - 1)
    elif order == 1:
        num_queries_per_step = 1
        num_sdf_gates = 1
    elif order == 2:
        num_queries_per_step = 2
        num_sdf_gates = 1

    return (num_queries_per_step * n_steps), (num_sdf_gates * n_steps)


def estimate_operator_norm_error(Ef, Eb, g, t, Lambda, order, n_steps):
    approximation = trotterized_unitary(
        Ef, Eb, g, t, Lambda, order=order, n_steps=n_steps
    )
    exact_evolution = trotterized_unitary(Ef, Eb, g, t, Lambda, order=0, n_steps=1)

    return (1 / t) * np.linalg.norm(exact_evolution - approximation, ord=2)


def estimate_state_fidelity(Ef, Eb, g, t, Lambda, order, n_steps, initial_state):
    approximation = trotterized_unitary(
        Ef, Eb, g, t, Lambda, order=order, n_steps=n_steps
    )
    exact_evolution = trotterized_unitary(Ef, Eb, g, t, Lambda, order=0, n_steps=1)

    return (
        np.abs(
            np.dot(
                (exact_evolution @ initial_state).T.conj(),
                (approximation @ initial_state),
            )
        )
        ** 2
    ).flatten()[0]


def max_qpe_time(Ef, Eb, g, Lambda):
    ham_op = (
        Ef * ParticleOperator("b0^ b0")
        + Eb * ParticleOperator("a0^ a0")
        + g
        * ParticleOperator("b0^ b0")
        * (ParticleOperator("a0^") + ParticleOperator("a0"))
    )
    ham_op_matrix = generate_matrix(ham_op, get_fock_basis(ham_op, Lambda), Lambda)
    return np.pi / np.linalg.norm(ham_op_matrix, ord=2)

import numpy as np
from itertools import combinations
from openparticle import Fock, ParticleOperator
from openparticle.utils import get_matrix_element


# gamma matrices
gamma0 = np.array([[0, 1], [1, 0]])
gamma1 = np.array([[0, -1], [1, 0]])
gammap = gamma0 + gamma1
gammam = gamma0 - gamma1
Lambdap = 0.5 * gamma0.dot(gammap)


def gamma_slash_minus_m(q, m):
    return gammap * (m**2 / q) + gammam * q - m * np.eye(2)


# discretized lightfront momentum
def p(k, L):
    return 2 * np.pi * k / L


# 1 + 1D lightcone spinors
def u(pk, m):
    return 1 / np.sqrt(np.abs(pk)) * np.array([pk, m]).reshape([-1, 1])


def v(pk, m):
    return 1 / np.sqrt(np.abs(pk)) * np.array([-pk, m]).reshape([-1, 1])


def ubar(pk, m):
    return (u(pk, m).reshape([1, -1])).dot(
        gamma0
    )  # u(p) is real so dagger -> transpose


def vbar(pk, m):
    return v(pk, m).reshape([1, -1]).dot(gamma0)  # v(p) is real so dagger -> transpose


def pminus(pplus, m):
    return m**2 / pplus


def pminuses(ppluses, ms):
    return [ms[i] ** 2 / ppluses[i] for i in range(len(ms))]


# Pplus operator
def Pplus(res):
    pp = ParticleOperator({})

    for n in np.arange(1 / 2, res + 1 / 2, 1):
        pp += n * (
            ParticleOperator("b" + str(int(n - 1 / 2)) + "^ ")
            * ParticleOperator("b" + str(int(n - 1 / 2)))
            + ParticleOperator("d" + str(int(n - 1 / 2)) + "^ ")
            * ParticleOperator("d" + str(int(n - 1 / 2)))
        )

    for m in range(1, res + 1):
        pp += m * (
            ParticleOperator("a" + str(m - 1) + "^ ")
            * ParticleOperator("a" + str(m - 1))
        )

    return pp


def Q(res):
    # Baryon number (Charge) operator

    B_op = ParticleOperator({})

    for n in range(0, res + 1, 1):
        B_op += ParticleOperator("b" + str(n) + "^ " + "b" + str(n)) - ParticleOperator(
            "d" + str(n) + "^ " + "d" + str(n)
        )

    return B_op


def impose_baryon_number(res, basis, baryon_number):

    baryon_number_basis = []
    for state in basis:
        if get_matrix_element(state, Q(res), state) == baryon_number:
            baryon_number_basis.append(state)

    return baryon_number_basis


# Partitioning Functions


def generate_fermion_antifermion_partitions(K):
    if K % 1 == 0:
        half_integers = [n / 2 for n in range(1, 2 * K + 1, 2)]  # [1/2, 3/2, 5/2, ...]
    else:
        half_integers = [n for n in np.arange(1 / 2, K + 1, 1)]
    valid_partitions = []

    for num_particles in range(0, len(half_integers) + 1):
        for config in combinations(half_integers, num_particles):
            particle_sum = sum(config)
            if particle_sum <= K:
                valid_partitions.append(config)
    return valid_partitions


def generate_boson_partitions(K):
    boson_partitions = []

    def partition_bosons(K, current_partition, start):
        if K == 0:
            boson_partitions.append(current_partition)
            return
        for boson in range(start, K + 1):
            partition_bosons(K - boson, current_partition + [boson], boson)

    partition_bosons(K, [], 1)
    return boson_partitions


def filter_partitions_by_n_particles(partitions, n_particles):
    for tup in partitions:
        total_count = sum(
            len(inner_tuple) for inner_tuple in tup
        )  # count n_particles in each partition
        if total_count <= n_particles:
            yield tup


def find_exact_partitions(K, n_particles: int = None):
    fermion_partitions = generate_fermion_antifermion_partitions(K)
    antifermion_partitions = generate_fermion_antifermion_partitions(K)
    all_partitions = []

    for fermions in fermion_partitions:
        for antifermions in antifermion_partitions:
            fermion_momentum = sum(fermions)
            antifermion_momentum = sum(antifermions)
            remaining_momentum = K - fermion_momentum - antifermion_momentum

            if remaining_momentum >= 0:
                boson_partitions = generate_boson_partitions(int(remaining_momentum))
                for bosons in boson_partitions:
                    total_momentum = (
                        fermion_momentum + antifermion_momentum + sum(bosons)
                    )
                    if total_momentum == K:
                        all_partitions.append((fermions, antifermions, bosons))

    if n_particles is not None:
        filtered_partitions = filter_partitions_by_n_particles(
            all_partitions, n_particles=n_particles
        )
        return filtered_partitions
    else:
        return all_partitions


def boson_partition_to_b_occ(b_partition):
    return list((i - 1, b_partition.count(i)) for i in list(set(b_partition)))


def fermion_partition_to_f_occ(f_tuple):
    return list(int(np.floor(i)) for i in f_tuple)


def momentum_states_partition(K, n_particles: int = None):
    partitions = find_exact_partitions(K, n_particles=n_particles)
    states = []
    for p in partitions:
        states.append(
            Fock(
                f_occ=fermion_partition_to_f_occ(p[0]),
                af_occ=fermion_partition_to_f_occ(p[1]),
                b_occ=boson_partition_to_b_occ(p[2]),
            )
        )

    return states


def get_Q_nonzero_sector_basis(resolution, n_fermions, n_bosons):
    partitions = find_exact_partitions(resolution)
    good_paritions = []

    for tup in partitions:
        n_fermions_in_partition = len(tup[0])
        n_antifermions_in_partition = len(tup[1])
        n_bosons_in_partition = len(tup[2])

        if (
            n_fermions_in_partition == n_fermions
            and n_bosons_in_partition == n_bosons
            and n_antifermions_in_partition == 0
        ):
            good_paritions.append(tup)

    states = []
    for p in good_paritions:
        states.append(
            Fock(
                f_occ=fermion_partition_to_f_occ(p[0]),
                af_occ=fermion_partition_to_f_occ(p[1]),
                b_occ=boson_partition_to_b_occ(p[2]),
            )
        )

    return states


def get_Q0_sector_basis(resolution, n_ffbar_pairs, n_bosons):
    partitions = find_exact_partitions(resolution)
    good_paritions = []

    for tup in partitions:
        n_fermions_in_partition = len(tup[0])
        n_antifermions_in_partition = len(tup[1])
        n_bosons_in_partition = len(tup[2])

        if (
            n_fermions_in_partition == n_antifermions_in_partition == n_ffbar_pairs
            and n_bosons_in_partition == n_bosons
        ):
            good_paritions.append(tup)

    states = []
    for p in good_paritions:
        states.append(
            Fock(
                f_occ=fermion_partition_to_f_occ(p[0]),
                af_occ=fermion_partition_to_f_occ(p[1]),
                b_occ=boson_partition_to_b_occ(p[2]),
            )
        )

    return states


def pdf(res, state, particle_type):
    # Calculates the parton distribution function for a given hadronic state

    particle_type_to_label = {"fermion": "b", "antifermion": "d", "boson": "a"}

    f = []
    x_fermion_arr = np.array([k / res for k in np.arange(1 / 2, res, 1)])
    x_boson_arr = np.array([k / res for k in np.arange(1, res + 1, 1)])

    if particle_type == "fermion" or "antifermion":
        for x in x_fermion_arr:
            pdf_op = ParticleOperator(
                particle_type_to_label[particle_type]
                + str(int(x * res - 1 / 2))
                + "^ "
                + particle_type_to_label[particle_type]
                + str(int(x * res - 1 / 2))
            )
            f.append(get_matrix_element(state, pdf_op, state))

    elif particle_type == "boson":
        for x in x_boson_arr:
            pdf_op = ParticleOperator(
                particle_type_to_label[particle_type]
                + str(int(x * res - 1))
                + "^ "
                + particle_type_to_label[particle_type]
                + str(int(x * res - 1))
            )
            f.append(get_matrix_element(state, pdf_op, state))

    return np.array(f)


def fock_sector_budget(eigenstate):

    budget = {}

    for state, coeff in eigenstate.state_dict.items():
        fock_state = Fock(state_dict={state: coeff})
        key = (fock_state.n_fermions, fock_state.n_antifermions, fock_state.n_bosons)
        budget[key] = np.abs(coeff) ** 2 + budget.get(key, 0)

    return budget

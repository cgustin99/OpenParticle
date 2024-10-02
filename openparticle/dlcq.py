import numpy as np
from itertools import combinations
from openparticle import Fock, ParticleOperator


# gamma matrices
gamma0 = np.array([[0, 1], [1, 0]])
gamma1 = np.array([[0, -1], [1, 0]])
gammap = gamma0 + gamma1
Lambdap = 0.5 * gamma0.dot(gammap)


# discretized lightfront momentum
def p(k, L=1):
    return 2 * np.pi * k / L


# 1 + 1D lightcone spinors
def u(k, m=1, L=1):
    return 1 / np.sqrt(np.abs(p(k, L))) * np.array([p(k, L), m]).reshape([-1, 1])


def v(k, m=1, L=1):
    return 1 / np.sqrt(np.abs(p(k, L))) * np.array([-p(k, L), m]).reshape([-1, 1])


def ubar(k, m=1, L=1):
    return (u(k, m, L).reshape([1, -1])).dot(
        gamma0
    )  # u(p) is real so dagger -> transpose


def vbar(k, m=1, L=1):
    return (
        v(k, m, L).reshape([1, -1]).dot(gamma0)
    )  # v(p) is real so dagger -> transpose


# Pplus operator
def Pplus(pp_cutoff):
    pp = ParticleOperator({})

    for n in range(1, pp_cutoff):
        pp += (n - 1 / 2) * (
            ParticleOperator("b" + str(n - 1) + "^")
            * ParticleOperator("b" + str(n - 1))
            + ParticleOperator("d" + str(n - 1) + "^")
            * ParticleOperator("d" + str(n - 1))
        )

    for m in range(1, pp_cutoff):
        pp += m * (
            ParticleOperator("a" + str(m - 1) + "^")
            * ParticleOperator("a" + str(m - 1))
        )

    return pp


# Partitioning Functions


def generate_fermion_antifermion_partitions(K):
    half_integers = [n / 2 for n in range(1, 2 * K + 1, 2)]  # [1/2, 3/2, 5/2, ...]
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


def find_exact_partitions(K):
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

    return all_partitions


def boson_partition_to_b_occ(b_partition):
    return list((i - 1, b_partition.count(i)) for i in list(set(b_partition)))


def fermion_partition_to_f_occ(f_tuple):
    return list(int(np.floor(i)) for i in f_tuple)


def Pplus_states_partition(K):
    partitions = find_exact_partitions(K)
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

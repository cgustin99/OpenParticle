from openparticle import *
import numpy as np
from typing import List
import itertools


def _fermion_combinations(n):
    if n is None:
        return []
    else:
        elements = list(range(n + 1))
        result = []
        for r in range(1, n + 2):  # Extend range to include the full list
            result.extend(itertools.combinations(elements, r))
        return [list(combo) for combo in result]


def _give_me_state(nn, Lambda, Omega):
    # Credit to Kamil Serafin
    n = nn
    result = list()
    for i in range(Lambda):
        r = n % (Omega + 1)
        n = n // (Omega + 1)
        result += [r]

    return list(enumerate(result))


def _boson_combinations(occupancy_cutoff, mode_cutoff):
    if mode_cutoff is None:
        return [[]]
    else:
        mode_cutoff += 1
        boson_combos = []
        for i in range((occupancy_cutoff + 1) ** mode_cutoff):
            boson_combos.append(_give_me_state(i, mode_cutoff, occupancy_cutoff))

        return boson_combos


def _all_combinations_of_lists(*lists):
    combinations = list(itertools.product(*lists))
    return [list(combo) for combo in combinations]


def _better_max(li):
    return max(li) if li else None


def get_fock_basis(operator: ParticleOperator, max_bose_occ: int = 1):
    # Main function that obtains the basis for the corresponding operator

    basis = []

    fermion_ops = []
    antifermion_ops = []
    boson_ops = []

    for op in operator.to_list():
        for decomposed_op in op.split():
            if isinstance(decomposed_op, FermionOperator):
                fermion_ops.append(decomposed_op)
            elif isinstance(decomposed_op, AntifermionOperator):
                antifermion_ops.append(decomposed_op)
            elif isinstance(decomposed_op, BosonOperator):
                boson_ops.append(decomposed_op)

    max_fermion_mode = _better_max([op.mode for op in fermion_ops])
    max_antifermion_mode = _better_max([op.mode for op in antifermion_ops])
    max_boson_mode = _better_max([op.mode for op in boson_ops])

    f_combs = _fermion_combinations(max_fermion_mode)
    af_combs = _fermion_combinations(max_antifermion_mode)
    b_combs = _boson_combinations(max_bose_occ, max_boson_mode)

    f_combs.insert(0, [])
    af_combs.insert(0, [])

    for i in _all_combinations_of_lists(f_combs, af_combs, b_combs):
        basis.append(Fock(f_occ=i[0], af_occ=i[1], b_occ=i[2]))

    return basis


def generate_matrix_from_basis(operator: ParticleOperator, basis: List[Fock]):
    size = (len(basis), len(basis))
    matrix = np.zeros(size)

    for i, state_i in enumerate(basis):
        for j, state_j in enumerate(basis):
            matrix_element = state_i.dagger() * operator * state_j
            matrix[i][j] = matrix_element

    return matrix

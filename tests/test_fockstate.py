from openparticle import Fock, FockSum, ConjugateFock, ConjugateFockSum
import pytest
import numpy as np


def test_fock_state_stores_correct_occupancies():
    f_occ = [2]  # One fermion in mode 2
    af_occ = [1, 3]  # One antifermion in mode 1, one in mode 3
    b_occ = [(0, 3)]  # Three bosons in mode 0
    coeff = 1.0

    fs = Fock(f_occ=f_occ, af_occ=af_occ, b_occ=b_occ)
    assert fs.f_occ == f_occ
    assert fs.af_occ == af_occ
    assert fs.b_occ == b_occ
    assert fs.coeff == coeff


@pytest.mark.parametrize("coeff", np.random.uniform(-100, 100, size=10))
def test_fock_state_mul_by_constant(coeff):
    f_occ = [2]  # One fermion in mode 2
    af_occ = [1, 3]  # One antifermion in mode 1, one in mode 3
    b_occ = [(0, 3)]  # Three bosons in mode 0

    fs = Fock(f_occ=f_occ, af_occ=af_occ, b_occ=b_occ)

    fs_new = coeff * fs

    assert fs_new.coeff == coeff * fs.coeff


def test_fock_states_add_two_different_states():
    state_1 = Fock([1], [2], [(0, 4)])
    state_2 = Fock([0], [], [(1, 1)])
    state_sum = state_1 + state_2
    # assert state_sum.__str__() == FockSum([state_1, state_2])
    assert state_sum.__str__() == "1.0 * |1; 2; (0, 4)⟩ + 1.0 * |0; ; (1, 1)⟩"


def test_fock_states_add_two_same_states():
    state = Fock([1], [2], [(0, 4)])
    # assert state + state == 2 * state
    assert (state + state).__str__() == "2.0 * |1; 2; (0, 4)⟩"


def test_add_fock_to_focksum():
    fsum = Fock([1], [2], []) + Fock([0], [], [])
    state = Fock([1], [], [])

    # assert fsum + state == FockSum([fsum.states_list[0], fsum.states_list[1], state])
    assert (fsum + state).__str__() == "1.0 * |1; 2; ⟩ + 1.0 * |0; ; ⟩ + 1.0 * |1; ; ⟩"

def test_add_focksum_to_fock_with_proper_ordering():
    fsum = Fock([1], [2], []) + Fock([0], [], [])
    state = Fock([1], [], [])

    # assert state + fsum == FockSum([state, fsum.states_list[0], fsum.states_list[1]])
    assert (state + fsum).__str__() == "1.0 * |1; ; ⟩ + 1.0 * |1; 2; ⟩ + 1.0 * |0; ; ⟩"

# ?????????
def test_add_focksum_to_fock_with_wrong_ordering():
    fsum = Fock([1], [2], []) + Fock([0], [], [])
    state = Fock([1], [], [])

    assert state + fsum == FockSum([fsum.states_list[0], fsum.states_list[1], state])
    assert (state + fsum).__str__() == "1.0 * |1; ; ⟩ + 1.0 * |1; 2; ⟩ + 1.0 * |0; ; ⟩"


def test_add_focksum_to_different_focksum():
    fsum1 = Fock([1], [2], []) + Fock([0], [], [])
    fsum2 = Fock([3], [2], []) + Fock([1], [], [])

    # assert fsum1 + fsum2 == FockSum(
    #     [
    #         fsum1.states_list[0],
    #         fsum1.states_list[1],
    #         fsum2.states_list[0],
    #         fsum2.states_list[1],
    #     ]
    # )

    assert (fsum1 + fsum2).__str__() == "1.0 * |1; 2; ⟩ + 1.0 * |0; ; ⟩ + 1.0 * |3; 2; ⟩ + 1.0 * |1; ; ⟩"


def test_add_focksum_to_overlapping_focksum():
    fsum1 = Fock([1], [2], []) + Fock([0], [], [])
    fsum2 = Fock([3], [2], []) + Fock([0], [], [])

    # assert fsum1 + fsum2 == Fock([1], [2], []) + 2 * Fock([0], [], []) + Fock(
    #     [3], [2], []
    # )
    assert (fsum1 + fsum2).__str__() == "1.0 * |1; 2; ⟩ + 2.0 * |0; ; ⟩ + 1.0 * |3; 2; ⟩"


def test_focksum_normlization():
    fsum = Fock([1], [2], []) + Fock([0], [], [])
    assert fsum.normalize() == 1 / np.sqrt(2) * fsum


def test_conjugate_fock_state_stores_correct_occupancies():
    f_occ = [2]  # One fermion in mode 2
    af_occ = [1, 3]  # One antifermion in mode 1, one in mode 3
    b_occ = [(0, 3)]  # Three bosons in mode 0
    coeff = 1.0

    fs = ConjugateFock(f_occ=f_occ, af_occ=af_occ, b_occ=b_occ)
    assert fs.f_occ == f_occ
    assert fs.af_occ == af_occ
    assert fs.b_occ == b_occ
    assert fs.coeff == coeff


@pytest.mark.parametrize("coeff", np.random.uniform(-100, 100, size=10))
def test_conjugate_fock_state_mul_by_constant(coeff):
    f_occ = [2]  # One fermion in mode 2
    af_occ = [1, 3]  # One antifermion in mode 1, one in mode 3
    b_occ = [(0, 3)]  # Three bosons in mode 0

    fs = ConjugateFock(f_occ=f_occ, af_occ=af_occ, b_occ=b_occ)

    fs_new = coeff * fs

    assert fs_new.coeff == coeff * fs.coeff


def test_conjugate_fock_states_add_two_different_states():
    state_1 = ConjugateFock([1], [2], [(0, 4)])
    state_2 = ConjugateFock([0], [], [(1, 1)])

    # assert state_1 + state_2 == ConjugateFockSum([state_1, state_2])
    assert (state_1 + state_2).__str__() == "1.0 * ⟨1; 2; (0, 4)| + 1.0 * ⟨0; ; (1, 1)|"


def test_conjugate_fock_states_add_two_same_states():
    state = ConjugateFock([1], [2], [(0, 4)])
    # assert state + state == 2 * state
    assert (state + state).__str__() == "2.0 * ⟨1; 2; (0, 4)|"


def test_add_conjugatefock_to_conjugatefocksum():
    fsum = ConjugateFock([1], [2], []) + ConjugateFock([0], [], [])
    state = ConjugateFock([1], [], [])

    # assert fsum + state == ConjugateFockSum(
    #     [fsum.states_list[0], fsum.states_list[1], state]
    # )
    assert (fsum + state).__str__() == "1.0 * ⟨1; 2; | + 1.0 * ⟨0; ; | + 1.0 * ⟨1; ; |"


def test_add_conjugatefocksum_to_conjugatefock_with_proper_ordering():
    fsum = ConjugateFock([1], [2], []) + ConjugateFock([0], [], [])
    state = ConjugateFock([1], [], [])

    # assert state + fsum == ConjugateFockSum(
    #     [state, fsum.states_list[0], fsum.states_list[1]]
    # )
    assert (fsum + state).__str__() == "1.0 * ⟨1; 2; | + 1.0 * ⟨0; ; | + 1.0 * ⟨1; ; |"


def test_add_conjugatefocksum_to_conjugatefock_with_wrong_ordering():
    fsum = ConjugateFock([1], [2], []) + ConjugateFock([0], [], [])
    state = ConjugateFock([1], [], [])

    assert state + fsum == ConjugateFockSum(
        [fsum.states_list[0], fsum.states_list[1], state]
    )


def test_add_conjugatefocksum_to_different_conjugatefocksum():
    fsum1 = ConjugateFock([1], [2], []) + ConjugateFock([0], [], [])
    fsum2 = ConjugateFock([3], [2], []) + ConjugateFock([1], [], [])

    # assert fsum1 + fsum2 == ConjugateFockSum(
    #     [
    #         fsum1.states_list[0],
    #         fsum1.states_list[1],
    #         fsum2.states_list[0],
    #         fsum2.states_list[1],
    #     ]
    # )

    assert (fsum1 + fsum2).__str__() == "1.0 * ⟨1; 2; | + 1.0 * ⟨0; ; | + 1.0 * ⟨3; 2; | + 1.0 * ⟨1; ; |"


def test_add_conjugatefocksum_to_overlapping_conjugatefocksum():
    fsum1 = ConjugateFock([1], [2], []) + ConjugateFock([0], [], [])
    fsum2 = ConjugateFock([3], [2], []) + ConjugateFock([0], [], [])

    # assert fsum1 + fsum2 == ConjugateFock([1], [2], []) + 2 * ConjugateFock(
    #     [0], [], []
    # ) + ConjugateFock([3], [2], [])

    assert (fsum1 + fsum2).__str__() == "1.0 * ⟨1; 2; | + 2.0 * ⟨0; ; | + 1.0 * ⟨3; 2; |"


def test_conjugatefocksum_normlization():
    fsum = ConjugateFock([1], [2], []) + ConjugateFock([0], [], [])
    assert fsum.normalize() == 1 / np.sqrt(2) * fsum


@pytest.mark.parametrize("f_occ", np.random.uniform(0, 100, size=2))
@pytest.mark.parametrize("af_occ", np.random.uniform(0, 100, size=2))
@pytest.mark.parametrize("b_occ_orbital", np.random.uniform(0, 100, size=2))
@pytest.mark.parametrize("b_occ_occupancy", np.random.uniform(0, 100, size=2))
@pytest.mark.parametrize("c_f_occ", np.random.uniform(0, 100, size=2))
@pytest.mark.parametrize("c_af_occ", np.random.uniform(0, 100, size=2))
@pytest.mark.parametrize("c_b_occ_orbital", np.random.uniform(0, 100, size=2))
@pytest.mark.parametrize("c_b_occ_occupancy", np.random.uniform(0, 100, size=2))
@pytest.mark.parametrize("f_coeff", np.random.uniform(-100, 100, size=2))
@pytest.mark.parametrize("c_coeff", np.random.uniform(-100, 100, size=2))
def test_inner_product(
    f_occ,
    af_occ,
    b_occ_orbital,
    b_occ_occupancy,
    c_f_occ,
    c_af_occ,
    c_b_occ_orbital,
    c_b_occ_occupancy,
    f_coeff,
    c_coeff,
):
    ket = f_coeff * Fock([f_occ], [af_occ], [(b_occ_orbital, b_occ_occupancy)])
    bra = c_coeff * ConjugateFock(
        [c_f_occ], [c_af_occ], [(c_b_occ_orbital, c_b_occ_occupancy)]
    )

    assert bra * ket == f_coeff * c_coeff * (
        f_occ
        == c_f_occ * af_occ
        == c_af_occ * b_occ_orbital
        == c_b_occ_orbital * b_occ_occupancy
        == c_b_occ_occupancy
    )

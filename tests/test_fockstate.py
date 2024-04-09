from openparticle.fockstate import FockState
import pytest
import numpy as np

def test_fock_state_stores_correct_occupancies():
    f_occ = [2] #One fermion in mode 2
    af_occ = [1, 3] #One antifermion in mode 1, one in mode 3
    b_occ = [(0, 3)] #Three bosons in mode 0
    coeff = 1.0

    fs = FockState(f_occ = f_occ, af_occ = af_occ, b_occ = b_occ)
    assert fs.f_occ == f_occ
    assert fs.af_occ == af_occ
    assert fs.b_occ == b_occ
    assert fs.coeff == coeff

@pytest.mark.parametrize("coeff", np.random.uniform(-100, 100, size=10))
def test_fock_state_mul_by_constant(coeff):
    f_occ = [2] #One fermion in mode 2
    af_occ = [1, 3] #One antifermion in mode 1, one in mode 3
    b_occ = [(0, 3)] #Three bosons in mode 0

    fs = FockState(f_occ = f_occ, af_occ = af_occ, b_occ = b_occ)

    fs_new = coeff * fs

    assert fs_new.coeff == coeff * fs.coeff



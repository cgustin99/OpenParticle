from pyscf import scf, gto, fci
from openfermionpyscf._run_pyscf import (
    compute_integrals,
    generate_molecular_hamiltonian,
)
from openfermion import InteractionOperator, FermionOperator
from openparticle import ParticleOperator, get_fock_basis, generate_matrix
import numpy as np
from functools import partial
import sys, os
import pytest


def get_nonzero_terms(arr, tol=1e-10):
    """ """
    where_nonzero = np.where(~np.isclose(arr, 0, atol=tol))
    nonzero_indices = list(zip(*where_nonzero))
    return dict(zip(nonzero_indices, arr[where_nonzero]))


@pytest.mark.parametrize(
    "molecule",
    [
        [("H", (0, 0, 0)), ("H", (0, 0, 1))],
        [("H", (0, 0, 0)), ("H", (0, 0, 1)), ("H", (0, 0, 2)), ("H", (0, 0, 3))],
    ],
)
def test_openparticle_molecule_energy(molecule):

    basis = "sto-3g"

    mol = gto.M(atom=molecule, basis=basis, unit="A").build()

    hf_obj = scf.RHF(mol)
    hf_obj.kernel()

    if 2 * mol.nao < 100:
        fci_obj = fci.FCI(hf_obj)
        fci_obj.kernel()

    one_electron_integrals, two_electron_integrals = compute_integrals(mol, hf_obj)

    hcore = np.zeros([mol.nao * 2] * 2)
    hcore[::2, ::2] = one_electron_integrals
    hcore[1::2, 1::2] = one_electron_integrals

    eri = np.zeros([mol.nao * 2] * 4)
    eri[::2, ::2, ::2, ::2] = two_electron_integrals
    eri[1::2, 1::2, 1::2, 1::2] = two_electron_integrals
    eri[0::2, 1::2, 1::2, 0::2] = two_electron_integrals
    eri[1::2, 0::2, 0::2, 1::2] = two_electron_integrals

    one_body_coefficients = get_nonzero_terms(hcore)
    two_body_coefficients = get_nonzero_terms(eri)

    fermionic_molecular_hamiltonian = FermionOperator()
    for (p, q), coeff in one_body_coefficients.items():
        fermionic_molecular_hamiltonian += FermionOperator(f"{p}^ {q}", coeff)
    for (p, q, r, s), coeff in two_body_coefficients.items():
        fermionic_molecular_hamiltonian += FermionOperator(
            f"{p}^ {q}^ {r} {s}", coeff * 0.5
        )

    op_fermionic_molecular_hamiltonian = ParticleOperator.from_openfermion(
        fermionic_molecular_hamiltonian
    )
    op_basis = get_fock_basis(op_fermionic_molecular_hamiltonian)
    matrix = generate_matrix(op_fermionic_molecular_hamiltonian, op_basis)

    vals = np.linalg.eigvalsh(matrix)

    assert np.allclose(min(vals) + hf_obj.energy_nuc(), fci_obj.e_tot)

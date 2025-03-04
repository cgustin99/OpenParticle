import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, CubicSpline
import time
import scipy


from openparticle.hamiltonians.free_hamiltonians import (
    free_boson_Hamiltonian,
    free_fermion_Hamiltonian,
)
from openparticle.hamiltonians.yukawa_hamiltonians import *
from openparticle.hamiltonians.renormalized_yukawa_hamiltonian import *
from openparticle.utils import *
from openparticle.dlcq import momentum_states_partition, pdf, fock_sector_budget


def get_eigenstates(
    resolutions,
    g,
    mf,
    mb,
    Q,
    num_eigenstates,
    n_particles: int = None,
    verbose: bool = False,
):
    eigenvalues = []
    eigenstates = []

    initial_time = time.time()
    for res in resolutions:
        if verbose:
            print("---K = ", res, "---")
        res_time_init = time.time()
        hamiltonian = yukawa_hamiltonian(res=res, g=g, mf=mf, mb=mb)
        if verbose:
            print("Size of Hamiltonian:", len(hamiltonian))
        if verbose:
            print("Hamiltonian generation:", time.time() - res_time_init)
        res_time_init = time.time()
        res_tmp_basis = momentum_states_partition(res, n_particles=n_particles)
        if verbose:
            print("Basis generation:", time.time() - res_time_init)
        res_time_init = time.time()
        baryon_number_basis = impose_baryon_number(res, res_tmp_basis, baryon_number=Q)
        res_time_init = time.time()
        if verbose:
            print("Size of Q+cutoff basis:", len(baryon_number_basis))

        if verbose:
            print("Matrix generation...")
        # tmp_mat = generate_matrix(hamiltonian, cutoff_basis)
        tmp_mat = generate_matrix_hermitian(hamiltonian, baryon_number_basis)
        if verbose:
            print("Matrix generation:", time.time() - res_time_init)
        res_time_init = time.time()

        # tmp_mat = generate_matrix(hamiltonian, baryon_number_basis)
        if tmp_mat.shape != (0, 0):
            if verbose:
                print("Calculating eigenvalues...")
            vals, vecs = np.linalg.eigh(tmp_mat)
            if verbose:
                print("Eigenvalues:", time.time() - res_time_init)
            vals = sorted(vals)
            # eigenvalues.append(vals[:num_eigenstates])
            # for i in range(0, num_eigenstates):
            #     eigenstates.append(numpy_to_fock(vecs[:, i], baryon_number_basis))

            if num_eigenstates != "all":
                eigenvalues.append(vals[:num_eigenstates])
                for i in range(0, num_eigenstates):
                    eigenstates.append(numpy_to_fock(vecs[:, i], baryon_number_basis))
            else:
                number_of_eigenstates = len(baryon_number_basis)
                eigenvalues.append(vals[:number_of_eigenstates])
                for i in range(0, number_of_eigenstates):
                    eigenstates.append(numpy_to_fock(vecs[:, i], baryon_number_basis))

    K_values = np.arange(1, len(eigenvalues) + 1)
    max_eigenvalues = max(len(eig) for eig in eigenvalues)

    padded_eigenvalues = np.array(
        [
            np.pad(eig, (0, max_eigenvalues - len(eig)), constant_values=np.nan)
            for eig in eigenvalues
        ]
    )

    padded_eigenvalues = padded_eigenvalues.T

    return padded_eigenvalues, eigenstates


def get_renormalized_eigenstates(
    resolutions,
    t,
    g,
    mf,
    mb,
    num_eigenstates,
    basis,
    n_particles: int = None,
    return_focks: bool = False,
    assert_positivity: bool = True,
    verbose: bool = False,
):
    eigenvalues = []
    eigenstates = []

    initial_time = time.time()
    for res in resolutions:
        if verbose:
            print("---K = ", res, "---")
        res_time_init = time.time()
        hamiltonian = renormalized_yukawa_hamiltonian(
            res=res, t=t, treg=0, g=g, mf=mf, mb=mb, verbose=verbose
        )
        if verbose:
            print("Size of Hamiltonian:", len(hamiltonian))
        if verbose:
            print("Hamiltonian generated in:", time.time() - res_time_init)
        # res_time_init = time.time()
        # res_tmp_basis = momentum_states_partition(res, n_particles=n_particles)
        # if verbose:
        #     print("Basis generated in:", time.time() - res_time_init)
        # res_time_init = time.time()
        # baryon_number_basis = impose_baryon_number(res, res_tmp_basis, baryon_number=Q)
        res_time_init = time.time()

        if verbose:
            print(
                "Generating",
                len(basis),
                "x",
                len(basis),
                "Matrix...",
            )
        # tmp_mat = generate_matrix(hamiltonian, cutoff_basis)
        tmp_mat = generate_matrix_hermitian(hamiltonian, basis)
        if verbose:
            print("Matrix generated in:", time.time() - res_time_init)
        res_time_init = time.time()

        # tmp_mat = generate_matrix(hamiltonian, baryon_number_basis)
        if tmp_mat.shape != (0, 0):
            if verbose:
                print("Calculating eigenvalues...")
            vals, vecs = np.linalg.eigh(tmp_mat)
            if verbose:
                print("Eigenvalues calculated in:", time.time() - res_time_init)
            vals = sorted(vals)

        start = time.time()
        if num_eigenstates != "all":
            eigenvalues.append(vals[:num_eigenstates])
            if return_focks:
                for i in range(0, num_eigenstates):
                    eigenstates.append(numpy_to_fock(vecs[:, i], basis))
        else:
            number_of_eigenstates = len(basis)
            eigenvalues.append(vals[:number_of_eigenstates])
            if return_focks:
                for i in range(0, number_of_eigenstates):
                    eigenstates.append(numpy_to_fock(vecs[:, i], basis))
        end = time.time()
        if verbose and return_focks:
            print("Time to 'Fockify':", end - start)

    max_eigenvalues = max(len(eig) for eig in eigenvalues)

    padded_eigenvalues = np.array(
        [
            np.pad(eig, (0, max_eigenvalues - len(eig)), constant_values=np.nan)
            for eig in eigenvalues
        ]
    )

    padded_eigenvalues = padded_eigenvalues.T
    if assert_positivity:
        assert all(padded_eigenvalues > 0)
    return padded_eigenvalues, eigenstates


def get_pdfs(
    resolution,
    renormalized_eigenvalues,
    renormalized_eigenvectors,
    num_eigenvectors,
    title: str = "",
):

    cols = 3
    rows = (num_eigenvectors + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3), dpi=200)
    axes = axes.flatten()

    handles, labels = [], []
    for idx, ax in enumerate(axes):
        if idx < num_eigenvectors:
            psi = renormalized_eigenvectors[idx]
            fermion_pdf = pdf(resolution, psi, "fermion")
            antifermion_pdf = pdf(resolution, psi, "antifermion")
            boson_pdf = pdf(resolution, psi, "boson")

            x_fermion = np.array(
                [k / resolution for k in np.arange(1 / 2, resolution, 1)]
            )
            x_boson = np.array(
                [k / resolution for k in np.arange(1, resolution + 1, 1)]
            )

            scatter_fermion = ax.scatter(
                x_fermion, np.real(fermion_pdf), marker=".", label="$f$"
            )

            if idx == 0:
                handles.extend([scatter_fermion])
                labels.extend(["$f$"])

            scatter_boson = ax.scatter(
                x_boson, np.real(boson_pdf), marker=".", label="$b$"
            )
            if idx == 0:
                handles.append(scatter_boson)
                labels.append("b")

            ax.set_xlabel("$x$")
            ax.set_ylabel("$f(x)$")
            # ax.set_yscale('log')
            ax.set_title(
                f"$M_{{{idx}}}^2 = {round(renormalized_eigenvalues[idx][0], 4)}$"
            )
        else:
            ax.axis("off")

    fig.suptitle(
        title,
        fontsize=16,
        y=1.02,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        fancybox=True,
        shadow=True,
        ncol=len(labels),
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()

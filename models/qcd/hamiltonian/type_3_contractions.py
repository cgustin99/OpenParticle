from openparticle.full_dlcq import *
from color_algebra import T, f
from gluon import *
from rgpep import *
import numpy as np
import numba as nb
from itertools import product

gluon_polarizations = np.array([1, -1])

spin_arrays = np.array(
    list(
        product(
            gluon_polarizations,
            gluon_polarizations,
            gluon_polarizations,
            gluon_polarizations,
        )
    ),
    dtype=np.int8,
)

color_combos = np.array(
    [
        [0, 1, 2, 1, 2],
        [0, 1, 2, 3, 6],
        [0, 1, 2, 4, 5],
        [0, 3, 6, 1, 2],
        [0, 3, 6, 3, 6],
        [0, 3, 6, 4, 5],
        [0, 4, 5, 1, 2],
        [0, 4, 5, 3, 6],
        [0, 4, 5, 4, 5],
        [1, 3, 5, 3, 5],
        [1, 3, 5, 4, 6],
        [1, 4, 6, 3, 5],
        [1, 4, 6, 4, 6],
        [2, 3, 4, 3, 4],
        [2, 3, 4, 5, 6],
        [2, 5, 6, 3, 4],
        [2, 5, 6, 5, 6],
        [3, 4, 7, 4, 7],
        [5, 6, 7, 5, 6],
    ],
    dtype=np.uint8,
)
fixed_qnums = np.vstack(
    [np.hstack([row1, spin]) for row1 in color_combos for spin in spin_arrays]
)


@nb.njit(
    ## output types
    nb.complex128[:, :, :, :, :, :, :, :, :, :]
    ##input types
    (nb.float64, nb.float64, nb.float64, nb.float64, nb.float64),
    ## other options
    fastmath=True,
    parallel=True,
)
def gluon_legs_term_tensor(K: float, Kp: float, t: float, g: float, mg: float = 1):

    boson_lim = int(K)
    boson_longitudinal_q = np.array(
        [i for i in np.arange(-boson_lim, boson_lim + 1, 1) if i != 0], dtype=np.float64
    )

    transverse_q = np.array([i for i in np.arange(-Kp, Kp + 1.0)], dtype=np.float64)

    dim = (  # particle 1
        len(boson_longitudinal_q),
        len(transverse_q),
        len(transverse_q),
        # particle 2
        len(boson_longitudinal_q),
        len(transverse_q),
        len(transverse_q),
        # particle 3
        len(boson_longitudinal_q),
        len(transverse_q),
        len(transverse_q),
        len(fixed_qnums),
    )
    gluon_legs_tensor = np.zeros(dim, dtype=np.complex128)

    for idx in nb.prange(len(boson_longitudinal_q)):
        q1plus = boson_longitudinal_q[idx]
        for q1perp1 in transverse_q:
            for q1perp2 in transverse_q:
                for q2plus in boson_longitudinal_q:
                    for q2perp1 in transverse_q:
                        for q2perp2 in transverse_q:
                            for q1primeplus in boson_longitudinal_q:
                                for q1primeperp1 in transverse_q:
                                    for q1primeperp2 in transverse_q:
                                        for (
                                            fixed_set_qnums_index,
                                            fixed_set_qnums,
                                        ) in enumerate(fixed_qnums):
                                            a, b, c = (
                                                fixed_set_qnums[0] + 1,
                                                fixed_set_qnums[1] + 1,
                                                fixed_set_qnums[2] + 1,
                                            )
                                            d, e = (
                                                fixed_set_qnums[3] + 1,
                                                fixed_set_qnums[4] + 1,
                                            )
                                            (
                                                polarization1,
                                                polarization2,
                                                polarization3,
                                                polarization4,
                                            ) = (
                                                fixed_set_qnums[5],
                                                fixed_set_qnums[6],
                                                fixed_set_qnums[7],
                                                fixed_set_qnums[8],
                                            )
                                            q1 = np.array(
                                                [q1plus, q1perp1, q1perp2],
                                                dtype=np.complex128,
                                            )
                                            q2 = np.array(
                                                [q2plus, q2perp1, q2perp2],
                                                dtype=np.complex128,
                                            )
                                            q1prime = np.array(
                                                [
                                                    q1primeplus,
                                                    q1primeperp1,
                                                    q1primeperp2,
                                                ],
                                                dtype=np.complex128,
                                            )
                                            q3 = -1 * (q1 + q2)  # delta(q1 + q2 + q3)
                                            q3prime = -q3  # delta(q3 + q3')
                                            q2prime = -1 * (
                                                q1prime - q3
                                            )  # delta(q1' + q2' + q3')

                                            if (
                                                0 < np.real(q3[0]) <= K
                                                and np.abs(q3[1]) <= Kp
                                                and np.abs(q3[2]) <= Kp
                                                and 0 < np.abs(q2prime[0]) <= K
                                                and np.abs(q2prime[1]) <= Kp
                                                and np.abs(q2prime[2]) <= Kp
                                                and 0 < np.abs(q3prime[0]) <= K
                                                and np.abs(q3prime[1]) <= Kp
                                                and np.abs(q3prime[2]) <= Kp
                                            ):
                                                A_1 = (
                                                    heaviside(q1[0])
                                                    * pol_vec(
                                                        p=q1,
                                                        pol=polarization1,
                                                    )
                                                    + heaviside(-q1[0])
                                                    * pol_vec(
                                                        p=-q1,
                                                        pol=polarization1,
                                                    ).conj()
                                                )
                                                A_2 = (
                                                    heaviside(q2[0])
                                                    * pol_vec(
                                                        p=q2,
                                                        pol=polarization2,
                                                    )
                                                    + heaviside(-q2[0])
                                                    * pol_vec(
                                                        p=-q2,
                                                        pol=polarization2,
                                                    ).conj()
                                                )
                                                A_1prime = (
                                                    heaviside(q1prime[0])
                                                    * pol_vec(
                                                        p=q1prime,
                                                        pol=polarization3,
                                                    )
                                                    + heaviside(-q1prime[0])
                                                    * pol_vec(
                                                        p=-q1prime,
                                                        pol=polarization3,
                                                    ).conj()
                                                )
                                                A_2prime = (
                                                    heaviside(q2prime[0])
                                                    * pol_vec(
                                                        p=q2prime,
                                                        pol=polarization4,
                                                    )
                                                    + heaviside(-q2prime[0])
                                                    * pol_vec(
                                                        p=-q2prime,
                                                        pol=polarization4,
                                                    ).conj()
                                                )
                                                coeff = 0
                                                for alpha in [1, 2, 3]:
                                                    if alpha == 1:
                                                        q_factor_alpha = (q3 - q2)[0]
                                                    else:
                                                        q_factor_alpha = (q3 - q2)[
                                                            alpha
                                                        ]
                                                    for beta in [1, 2, 3]:
                                                        if beta == 1:
                                                            q_factor_beta = (q1 - q3)[0]
                                                        else:
                                                            q_factor_beta = (q1 - q3)[
                                                                beta
                                                            ]
                                                        for delta in [1, 2, 3]:
                                                            if delta == 1:
                                                                q_factor_delta = (
                                                                    q3prime - q2prime
                                                                )[0]
                                                            else:
                                                                q_factor_delta = (
                                                                    q3prime - q2prime
                                                                )[delta]
                                                            for omega in [1, 2, 3]:
                                                                if omega == 1:
                                                                    q_factor_omega = (
                                                                        q1prime
                                                                        - q3prime
                                                                    )[0]
                                                                else:
                                                                    q_factor_omega = (
                                                                        q1prime
                                                                        - q3prime
                                                                    )[omega]
                                                                for mu in [1, 2, 3]:
                                                                    if mu == 1:
                                                                        q_factor_mu = (
                                                                            q2 - q1
                                                                        )[0]
                                                                    else:
                                                                        q_factor_mu = (
                                                                            q2 - q1
                                                                        )[mu]
                                                                    for nu in [1, 2, 3]:
                                                                        if nu == 1:
                                                                            q_factor_nu = (
                                                                                q2prime
                                                                                - q1prime
                                                                            )[
                                                                                0
                                                                            ]
                                                                        else:
                                                                            q_factor_nu = (
                                                                                q2prime
                                                                                - q1prime
                                                                            )[
                                                                                nu
                                                                            ]
                                                                        coeff += (
                                                                            (
                                                                                q_factor_mu
                                                                                * g_lower_indices[
                                                                                    alpha,
                                                                                    beta,
                                                                                ]
                                                                                + q_factor_alpha
                                                                                * g_lower_indices[
                                                                                    beta,
                                                                                    mu,
                                                                                ]
                                                                                + q_factor_beta
                                                                                * g_lower_indices[
                                                                                    mu,
                                                                                    alpha,
                                                                                ]
                                                                            )
                                                                            * (
                                                                                q_factor_nu
                                                                                * g_lower_indices[
                                                                                    delta,
                                                                                    omega,
                                                                                ]
                                                                                + q_factor_delta
                                                                                * g_lower_indices[
                                                                                    omega,
                                                                                    nu,
                                                                                ]
                                                                                + q_factor_omega
                                                                                * g_lower_indices[
                                                                                    nu,
                                                                                    delta,
                                                                                ]
                                                                            )
                                                                            * A_1[alpha]
                                                                            * A_2[beta]
                                                                            * A_1prime[
                                                                                delta
                                                                            ]
                                                                            * A_2prime[
                                                                                omega
                                                                            ]
                                                                        )
                                                q1minus = (
                                                    mg**2 + q1[1:3].dot(q1[1:3])
                                                ) / q1[0]
                                                q2minus = (
                                                    mg**2 + q2[1:3].dot(q2[1:3])
                                                ) / q2[0]
                                                q3minus = (
                                                    mg**2 + q3[1:3].dot(q3[1:3])
                                                ) / q3[0]
                                                q1primeminus = (
                                                    mg**2
                                                    + q1prime[1:3].dot(q1prime[1:3])
                                                ) / q1prime[0]
                                                q2primeminus = (
                                                    mg**2
                                                    + q2prime[1:3].dot(q2prime[1:3])
                                                ) / q2prime[0]
                                                q3primeminus = (
                                                    mg**2
                                                    + q3prime[1:3].dot(q3prime[1:3])
                                                ) / q3prime[0]

                                                Q = q1minus + q2minus + q3minus
                                                Qp = (
                                                    q1primeminus
                                                    + q2primeminus
                                                    + q3primeminus
                                                )
                                                coeff *= (
                                                    g**2
                                                    * f(a, b, c)
                                                    * f(d, e, c)
                                                    * 1
                                                    / (
                                                        np.abs(q1[0])  # From fields
                                                        * np.abs(q2[0])  # From fields
                                                        * np.abs(
                                                            q3[0]
                                                        )  # From contraction
                                                        * np.abs(
                                                            q1prime[0]
                                                        )  # From fields
                                                        * np.abs(
                                                            q2prime[0]
                                                        )  # From fields
                                                    )
                                                    * rgpep_factor_second_order(
                                                        Q, Qp, t
                                                    )
                                                )
                                                if coeff != 0:

                                                    q1plus_index = np.where(
                                                        boson_longitudinal_q == q1plus
                                                    )[0][0]
                                                    q1perp1_index = np.where(
                                                        transverse_q == q1perp1
                                                    )[0][0]
                                                    q1perp2_index = np.where(
                                                        transverse_q == q1perp2
                                                    )[0][0]

                                                    q2plus_index = np.where(
                                                        boson_longitudinal_q == q2plus
                                                    )[0][0]
                                                    q2perp1_index = np.where(
                                                        transverse_q == q2perp1
                                                    )[0][0]
                                                    q2perp2_index = np.where(
                                                        transverse_q == q2perp2
                                                    )[0][0]

                                                    q3plus_index = np.where(
                                                        boson_longitudinal_q == q3[0]
                                                    )[0][0]
                                                    q3perp1_index = np.where(
                                                        transverse_q == q3[1]
                                                    )[0][0]
                                                    q3perp2_index = np.where(
                                                        transverse_q == q3[2]
                                                    )[0][0]

                                                    gluon_legs_tensor[
                                                        q1plus_index,
                                                        q1perp1_index,
                                                        q1perp2_index,
                                                        q2plus_index,
                                                        q2perp1_index,
                                                        q2perp2_index,
                                                        q3plus_index,
                                                        q3perp1_index,
                                                        q3perp2_index,
                                                        fixed_set_qnums_index,
                                                    ] = coeff

    return gluon_legs_tensor

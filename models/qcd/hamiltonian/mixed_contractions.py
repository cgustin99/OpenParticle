from openparticle.full_dlcq import *
from color_algebra import T, f
from gluon import *
from rgpep import *
import numpy as np
import numba as nb
from itertools import product

gluon_polarizations = np.array([1, -1])
quark_helicities = np.array([1, -1])

spin_arrays = np.array(
    list(
        product(
            quark_helicities,
            quark_helicities,
            gluon_polarizations,
            gluon_polarizations,
        )
    ),
    dtype=np.int8,
)

color_combos = np.array(
    [
        [0, 0, 2, 3, 4],
        [0, 0, 2, 5, 6],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 3, 6],
        [0, 1, 0, 4, 5],
        [0, 1, 1, 3, 5],
        [0, 1, 1, 4, 6],
        [0, 2, 3, 4, 7],
        [1, 0, 0, 1, 2],
        [1, 0, 0, 3, 6],
        [1, 0, 0, 4, 5],
        [1, 0, 1, 3, 5],
        [1, 0, 1, 4, 6],
        [1, 1, 2, 3, 4],
        [1, 1, 2, 5, 6],
        [1, 2, 5, 6, 7],
        [2, 0, 3, 4, 7],
        [2, 1, 5, 6, 7],
    ]
)
fixed_qnums = np.vstack(
    [np.hstack([row1, spin]) for row1 in color_combos for spin in spin_arrays]
)


@nb.njit(
    ## output types
    nb.complex128[:, :, :, :, :, :, :, :, :, :]
    ##input types
    (nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64),
    ## other options
    fastmath=True,
    parallel=True,
)
def effective_gluon_exchange_mixed_legs_term_tensor(
    K: float,
    Kp: float,
    s: float,
    g: float,
    mq: float,
    mg: float,
):

    fermion_lim = K - ((K - 0.5) % 1)
    boson_lim = int(K)
    fermion_longitudinal_q = np.array(
        [i for i in np.arange(-fermion_lim, fermion_lim + 1.0, 1.0) if i != 0.0],
        dtype=np.float64,
    )
    boson_longitudinal_q = np.array(
        [i for i in np.arange(-boson_lim, boson_lim + 1.0, 1.0) if i != 0.0],
        dtype=np.float64,
    )

    transverse_q = np.array([i for i in np.arange(-Kp, Kp + 1.0)], dtype=np.float64)

    dim = (  # particle 1
        len(fermion_longitudinal_q),
        len(transverse_q),
        len(transverse_q),
        # particle 2
        len(fermion_longitudinal_q),
        len(transverse_q),
        len(transverse_q),
        # particle 1prime
        len(boson_longitudinal_q),
        len(transverse_q),
        len(transverse_q),
        len(fixed_qnums),
    )
    gluon_exch_tensor = np.zeros(dim, dtype=np.complex128)

    for idx in nb.prange(len(fermion_longitudinal_q)):
        q1plus = fermion_longitudinal_q[idx]
        for q1perp1 in transverse_q:
            for q1perp2 in transverse_q:
                for q2plus in fermion_longitudinal_q:
                    for q2perp1 in transverse_q:
                        for q2perp2 in transverse_q:
                            for q1primeplus in boson_longitudinal_q:
                                for q1primeperp1 in transverse_q:
                                    for q1primeperp2 in transverse_q:
                                        for (
                                            fixed_set_qnums_index,
                                            fixed_set_qnums,
                                        ) in enumerate(fixed_qnums):
                                            c1, c2, a = (
                                                fixed_set_qnums[0],
                                                fixed_set_qnums[1],
                                                fixed_set_qnums[2] + 1,
                                            )
                                            b, c = (
                                                fixed_set_qnums[3] + 1,
                                                fixed_set_qnums[4] + 1,
                                            )
                                            (
                                                helicity1,
                                                helicity2,
                                                polarization1,
                                                polarization2,
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
                                                0 < np.abs(q3[0]) <= K
                                                and np.abs(q3[1]) <= Kp
                                                and np.abs(q3[2]) <= Kp
                                                and 0 < np.abs(q2prime[0]) <= K
                                                and np.abs(q2prime[1]) <= Kp
                                                and np.abs(q2prime[2]) <= Kp
                                                and 0 < np.abs(q3prime[0]) <= K
                                                and np.abs(q3prime[1]) <= Kp
                                                and np.abs(q3prime[2]) <= Kp
                                            ):

                                                psi_1 = heaviside(-q1[0]) * udag(
                                                    p=-q1, m=mq, h=helicity1
                                                ) + heaviside(q1[0]) * vdag(
                                                    p=q1, m=mq, h=helicity1
                                                )
                                                psi_2 = heaviside(q2[0]) * u(
                                                    p=q2, m=mq, h=helicity2
                                                ) + heaviside(-q2[0]) * v(
                                                    p=-q2, m=mq, h=helicity2
                                                )
                                                A_1p = (
                                                    heaviside(q1prime[0])
                                                    * pol_vec(
                                                        p=q1prime,
                                                        pol=polarization1,
                                                    )
                                                    + heaviside(-q1prime[0])
                                                    * pol_vec(
                                                        p=-q1prime,
                                                        pol=polarization1,
                                                    ).conj()
                                                )
                                                A_2p = (
                                                    heaviside(q2prime[0])
                                                    * pol_vec(
                                                        p=q2prime,
                                                        pol=polarization1,
                                                    )
                                                    + heaviside(-q2prime[0])
                                                    * pol_vec(
                                                        p=-q2prime,
                                                        pol=polarization1,
                                                    ).conj()
                                                )

                                                coeff = 0
                                                for mu in [0, 1, 2, 3]:
                                                    for gamma in [1, 2, 3]:
                                                        if (
                                                            gamma == 1
                                                        ):  # (q2' - q1')_minus = 0.5 * (q2' - q1')^plus
                                                            q_factor_gamma = (
                                                                0.5
                                                                * (q2prime - q1prime)[
                                                                    gamma
                                                                ]
                                                            )
                                                        else:
                                                            q_factor_gamma = (
                                                                q2prime - q1prime
                                                            )[
                                                                gamma
                                                            ]  # (q2' - q1')_perp = (q2' - q1')^perp
                                                        for alpha in [1, 2, 3]:
                                                            if alpha == 1:
                                                                q_factor_alpha = (
                                                                    0.5
                                                                    * (
                                                                        q3prime
                                                                        - q2prime
                                                                    )[0]
                                                                )
                                                            else:
                                                                q_factor_alpha = (
                                                                    q3prime - q2prime
                                                                )[alpha]
                                                            for beta in [1, 2, 3]:
                                                                if beta == 1:
                                                                    q_factor_beta = (
                                                                        0.5
                                                                        * (
                                                                            q1prime
                                                                            - q2prime
                                                                        )[0]
                                                                    )
                                                                else:
                                                                    q_factor_beta = (
                                                                        q1prime
                                                                        - q2prime
                                                                    )[beta]
                                                                coeff += d(
                                                                    q3,
                                                                    mg,
                                                                    mu,
                                                                    gamma,
                                                                    "upper",
                                                                ) * (
                                                                    q_factor_gamma
                                                                    * g_lower_indices[
                                                                        alpha, beta
                                                                    ]
                                                                    + q_factor_alpha
                                                                    * g_lower_indices[
                                                                        beta, gamma
                                                                    ]
                                                                    + q_factor_beta
                                                                    * g_lower_indices[
                                                                        gamma, alpha
                                                                    ]
                                                                )
                                                    coeff += psi_1.dot(
                                                        gamma0.dot(
                                                            gamma_lower[mu].dot(psi_2)
                                                        )
                                                        * A_1p[alpha]
                                                        * A_2p[beta]
                                                    )

                                                q1minus = (
                                                    mq**2 + q1[1:3].dot(q1[1:3])
                                                ) / q1[0]
                                                q2minus = (
                                                    mq**2 + q2[1:3].dot(q2[1:3])
                                                ) / q2[0]
                                                q3minus = (
                                                    mg**2 + q3[1:3].dot(q3[1:3])
                                                ) / q3[0]
                                                q1primeminus = (
                                                    mq**2
                                                    + q1prime[1:3].dot(q1prime[1:3])
                                                ) / q1prime[0]
                                                q2primeminus = (
                                                    mq**2
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
                                                    * T[a, c1, c2]
                                                    * f(a, b, c)
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
                                                        Q, Qp, s
                                                    )
                                                )
                                                if coeff != 0:

                                                    q1plus_index = np.where(
                                                        fermion_longitudinal_q == q1plus
                                                    )[0][0]
                                                    q1perp1_index = np.where(
                                                        transverse_q == q1perp1
                                                    )[0][0]
                                                    q1perp2_index = np.where(
                                                        transverse_q == q1perp2
                                                    )[0][0]

                                                    q2plus_index = np.where(
                                                        fermion_longitudinal_q == q2plus
                                                    )[0][0]
                                                    q2perp1_index = np.where(
                                                        transverse_q == q2perp1
                                                    )[0][0]
                                                    q2perp2_index = np.where(
                                                        transverse_q == q2perp2
                                                    )[0][0]

                                                    q1primeplus_index = np.where(
                                                        fermion_longitudinal_q
                                                        == q1prime[0]
                                                    )[0][0]
                                                    q1primeperp1_index = np.where(
                                                        transverse_q == q1prime[1]
                                                    )[0][0]
                                                    q1primeperp2_index = np.where(
                                                        transverse_q == q1prime[2]
                                                    )[0][0]

                                                    gluon_exch_tensor[
                                                        q1plus_index,
                                                        q1perp1_index,
                                                        q1perp2_index,
                                                        q2plus_index,
                                                        q2perp1_index,
                                                        q2perp2_index,
                                                        q1primeplus_index,
                                                        q1primeperp1_index,
                                                        q1primeperp2_index,
                                                        fixed_set_qnums_index,
                                                    ] = coeff

    return gluon_exch_tensor

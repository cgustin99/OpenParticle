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


# Requires counterterm
@nb.njit(
    ## output types
    nb.complex128[:, :, :, :, :, :, :, :, :, :]
    ##input types
    (nb.float64, nb.float64, nb.float64, nb.float64, nb.float64),
    ## other options
    fastmath=True,
    parallel=True,
)
def effective_gluon_legs_term_tensor(
    K: float, Kp: float, s: float, g: float, mg: float = 1
):

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
                                                0 <= np.real(q3[0]) <= K
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
                                                coeff = 0

                                                if q3[0] != 0:
                                                    for gamma in [1, 2, 3]:
                                                        for alpha in [1, 2, 3]:
                                                            for beta in [1, 2, 3]:
                                                                if gamma == 1:
                                                                    qfactor_gamma = (
                                                                        1
                                                                        / 2
                                                                        * (q2 - q1)[0]
                                                                    )
                                                                else:
                                                                    qfactor_gamma = (
                                                                        -1
                                                                        * (q2 - q1)[
                                                                            gamma
                                                                        ]
                                                                    )
                                                                if alpha == 1:
                                                                    qfactor_alpha = (
                                                                        1
                                                                        / 2
                                                                        * (q3 - q2)[0]
                                                                    )
                                                                else:
                                                                    qfactor_alpha = (
                                                                        -1
                                                                        * (q3 - q2)[
                                                                            alpha
                                                                        ]
                                                                    )
                                                                if beta == 1:
                                                                    qfactor_beta = (
                                                                        1
                                                                        / 2
                                                                        * (q1 - q3)[0]
                                                                    )
                                                                else:
                                                                    qfactor_beta = (
                                                                        -1
                                                                        * (q1 - q3)[
                                                                            beta
                                                                        ]
                                                                    )
                                                                F = (
                                                                    1j
                                                                    * f(a, b, c)
                                                                    * qfactor_gamma
                                                                    * g_lower_indices[
                                                                        alpha, beta
                                                                    ]
                                                                    * qfactor_alpha
                                                                    * g_lower_indices[
                                                                        beta, gamma
                                                                    ]
                                                                    * qfactor_beta
                                                                    * g_lower_indices[
                                                                        gamma, alpha
                                                                    ]
                                                                )
                                                                for sigma in [1, 2, 3]:
                                                                    for delta in [
                                                                        1,
                                                                        2,
                                                                        3,
                                                                    ]:
                                                                        for omega in [
                                                                            1,
                                                                            2,
                                                                            3,
                                                                        ]:
                                                                            if (
                                                                                sigma
                                                                                == 1
                                                                            ):
                                                                                qfactor_sigma = (
                                                                                    1
                                                                                    / 2
                                                                                    * (
                                                                                        q2prime
                                                                                        - q1prime
                                                                                    )[0]
                                                                                )
                                                                            else:
                                                                                qfactor_sigma = (
                                                                                    -1
                                                                                    * (
                                                                                        q2prime
                                                                                        - q1prime
                                                                                    )[
                                                                                        sigma
                                                                                    ]
                                                                                )
                                                                            if (
                                                                                delta
                                                                                == 1
                                                                            ):
                                                                                qfactor_delta = (
                                                                                    1
                                                                                    / 2
                                                                                    * (
                                                                                        q3prime
                                                                                        - q2prime
                                                                                    )[0]
                                                                                )
                                                                            else:
                                                                                qfactor_delta = (
                                                                                    -1
                                                                                    * (
                                                                                        q3prime
                                                                                        - q2prime
                                                                                    )[
                                                                                        delta
                                                                                    ]
                                                                                )
                                                                            if (
                                                                                omega
                                                                                == 1
                                                                            ):
                                                                                qfactor_omega = (
                                                                                    1
                                                                                    / 2
                                                                                    * (
                                                                                        q1prime
                                                                                        - q3prime
                                                                                    )[0]
                                                                                )
                                                                            else:
                                                                                qfactor_omega = (
                                                                                    -1
                                                                                    * (
                                                                                        q1prime
                                                                                        - q3prime
                                                                                    )[
                                                                                        omega
                                                                                    ]
                                                                                )
                                                                            Fprime = (
                                                                                1j
                                                                                * f(
                                                                                    a,
                                                                                    d,
                                                                                    e,
                                                                                )
                                                                                * qfactor_sigma
                                                                                * g_lower_indices[
                                                                                    delta,
                                                                                    omega,
                                                                                ]
                                                                                * qfactor_delta
                                                                                * g_lower_indices[
                                                                                    omega,
                                                                                    sigma,
                                                                                ]
                                                                                * qfactor_omega
                                                                                * g_lower_indices[
                                                                                    sigma,
                                                                                    delta,
                                                                                ]
                                                                            )
                                                                            for mu in [
                                                                                0,
                                                                                1,
                                                                                2,
                                                                                3,
                                                                            ]:
                                                                                for (
                                                                                    nu
                                                                                ) in [
                                                                                    0,
                                                                                    1,
                                                                                    2,
                                                                                    3,
                                                                                ]:

                                                                                    coeff += (
                                                                                        -(
                                                                                            g
                                                                                            ** 2
                                                                                        )
                                                                                        / 2
                                                                                        * (
                                                                                            (
                                                                                                rgpep_factor_first_order(
                                                                                                    Q,
                                                                                                    s,
                                                                                                )
                                                                                                * rgpep_factor_first_order(
                                                                                                    Qp,
                                                                                                    s,
                                                                                                )
                                                                                                - rgpep_factor_first_order(
                                                                                                    Q
                                                                                                    + Qp,
                                                                                                    s,
                                                                                                )
                                                                                            )
                                                                                            * 1
                                                                                            / 2
                                                                                            * (
                                                                                                1
                                                                                                / Q
                                                                                                - 1
                                                                                                / Qp
                                                                                            )
                                                                                            * g_lower_indices[
                                                                                                mu,
                                                                                                nu,
                                                                                            ]
                                                                                            * (
                                                                                                g_upper_indices[
                                                                                                    mu,
                                                                                                    gamma,
                                                                                                ]
                                                                                                * F
                                                                                                * (
                                                                                                    A_1[
                                                                                                        alpha
                                                                                                    ]
                                                                                                    * A_2[
                                                                                                        beta
                                                                                                    ]
                                                                                                )
                                                                                                * g_upper_indices[
                                                                                                    nu,
                                                                                                    sigma,
                                                                                                ]
                                                                                                * Fprime
                                                                                                * A_1prime[
                                                                                                    delta
                                                                                                ]
                                                                                                * A_2prime[
                                                                                                    omega
                                                                                                ]
                                                                                            )
                                                                                        )
                                                                                    )
                                                                            coeff += (
                                                                                -(g**2)
                                                                                / 2
                                                                                * (
                                                                                    (
                                                                                        rgpep_factor_first_order(
                                                                                            Q,
                                                                                            s,
                                                                                        )
                                                                                        * rgpep_factor_first_order(
                                                                                            Qp,
                                                                                            s,
                                                                                        )
                                                                                    )
                                                                                    * 1
                                                                                    / 2
                                                                                    * (
                                                                                        1
                                                                                        / Q
                                                                                        - 1
                                                                                        / Qp
                                                                                    )
                                                                                    * (
                                                                                        (
                                                                                            (
                                                                                                q1
                                                                                                + q2
                                                                                            ).dot(
                                                                                                q1
                                                                                                + q2
                                                                                            )
                                                                                            + (
                                                                                                q1prime
                                                                                                + q2prime
                                                                                            ).dot(
                                                                                                q1prime
                                                                                                + q2prime
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                    / (
                                                                                        2
                                                                                        * (
                                                                                            q3[
                                                                                                0
                                                                                            ]
                                                                                            ** 2
                                                                                        )
                                                                                    )
                                                                                    * g_lower_indices[
                                                                                        0,
                                                                                        0,
                                                                                    ]
                                                                                    * (
                                                                                        g_upper_indices[
                                                                                            mu,
                                                                                            gamma,
                                                                                        ]
                                                                                        * F
                                                                                        * (
                                                                                            A_1[
                                                                                                alpha
                                                                                            ]
                                                                                            * A_2[
                                                                                                beta
                                                                                            ]
                                                                                        )
                                                                                        * g_upper_indices[
                                                                                            nu,
                                                                                            sigma,
                                                                                        ]
                                                                                        * Fprime
                                                                                        * A_1prime[
                                                                                            delta
                                                                                        ]
                                                                                        * A_2prime[
                                                                                            omega
                                                                                        ]
                                                                                    )
                                                                                )
                                                                            )
                                                                            coeff += (
                                                                                -(g**2)
                                                                                / 2
                                                                                * (
                                                                                    (
                                                                                        rgpep_factor_first_order(
                                                                                            Q
                                                                                            + Qp,
                                                                                            s,
                                                                                        )
                                                                                    )
                                                                                    * 1
                                                                                    / 2
                                                                                    * (
                                                                                        1
                                                                                        / Q
                                                                                        - 1
                                                                                        / Qp
                                                                                    )
                                                                                    * (
                                                                                        (
                                                                                            (
                                                                                                q1
                                                                                                + q2
                                                                                            ).dot(
                                                                                                q1
                                                                                                + q2
                                                                                            )
                                                                                            + (
                                                                                                q1prime
                                                                                                + q2prime
                                                                                            ).dot(
                                                                                                q1prime
                                                                                                + q2prime
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                    / (
                                                                                        2
                                                                                        * (
                                                                                            q3[
                                                                                                0
                                                                                            ]
                                                                                            ** 2
                                                                                            - 1
                                                                                            / (
                                                                                                4
                                                                                                * K
                                                                                                ** 2
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                    * g_lower_indices[
                                                                                        0,
                                                                                        0,
                                                                                    ]
                                                                                    * (
                                                                                        g_upper_indices[
                                                                                            mu,
                                                                                            gamma,
                                                                                        ]
                                                                                        * F
                                                                                        * (
                                                                                            A_1[
                                                                                                alpha
                                                                                            ]
                                                                                            * A_2[
                                                                                                beta
                                                                                            ]
                                                                                        )
                                                                                        * g_upper_indices[
                                                                                            nu,
                                                                                            sigma,
                                                                                        ]
                                                                                        * Fprime
                                                                                        * A_1prime[
                                                                                            delta
                                                                                        ]
                                                                                        * A_2prime[
                                                                                            omega
                                                                                        ]
                                                                                    )
                                                                                )
                                                                            )
                                                elif q3[0] == 0:
                                                    for gamma in [1, 2, 3]:
                                                        for alpha in [1, 2, 3]:
                                                            for beta in [1, 2, 3]:
                                                                if gamma == 1:
                                                                    qfactor_gamma = (
                                                                        1
                                                                        / 2
                                                                        * (q2 - q1)[0]
                                                                    )
                                                                else:
                                                                    qfactor_gamma = (
                                                                        -1
                                                                        * (q2 - q1)[
                                                                            gamma
                                                                        ]
                                                                    )
                                                                if alpha == 1:
                                                                    qfactor_alpha = (
                                                                        1
                                                                        / 2
                                                                        * (q3 - q2)[0]
                                                                    )
                                                                else:
                                                                    qfactor_alpha = (
                                                                        -1
                                                                        * (q3 - q2)[
                                                                            alpha
                                                                        ]
                                                                    )
                                                                if beta == 1:
                                                                    qfactor_beta = (
                                                                        1
                                                                        / 2
                                                                        * (q1 - q3)[0]
                                                                    )
                                                                else:
                                                                    qfactor_beta = (
                                                                        -1
                                                                        * (q1 - q3)[
                                                                            beta
                                                                        ]
                                                                    )
                                                                F = (
                                                                    1j
                                                                    * f(a, b, c)
                                                                    * qfactor_gamma
                                                                    * g_lower_indices[
                                                                        alpha, beta
                                                                    ]
                                                                    * qfactor_alpha
                                                                    * g_lower_indices[
                                                                        beta, gamma
                                                                    ]
                                                                    * qfactor_beta
                                                                    * g_lower_indices[
                                                                        gamma, alpha
                                                                    ]
                                                                )
                                                                for sigma in [1, 2, 3]:
                                                                    for delta in [
                                                                        1,
                                                                        2,
                                                                        3,
                                                                    ]:
                                                                        for omega in [
                                                                            1,
                                                                            2,
                                                                            3,
                                                                        ]:
                                                                            if (
                                                                                sigma
                                                                                == 1
                                                                            ):
                                                                                qfactor_sigma = (
                                                                                    1
                                                                                    / 2
                                                                                    * (
                                                                                        q2prime
                                                                                        - q1prime
                                                                                    )[0]
                                                                                )
                                                                            else:
                                                                                qfactor_sigma = (
                                                                                    -1
                                                                                    * (
                                                                                        q2prime
                                                                                        - q1prime
                                                                                    )[
                                                                                        sigma
                                                                                    ]
                                                                                )
                                                                            if (
                                                                                delta
                                                                                == 1
                                                                            ):
                                                                                qfactor_delta = (
                                                                                    1
                                                                                    / 2
                                                                                    * (
                                                                                        q3prime
                                                                                        - q2prime
                                                                                    )[0]
                                                                                )
                                                                            else:
                                                                                qfactor_delta = (
                                                                                    -1
                                                                                    * (
                                                                                        q3prime
                                                                                        - q2prime
                                                                                    )[
                                                                                        delta
                                                                                    ]
                                                                                )
                                                                            if (
                                                                                omega
                                                                                == 1
                                                                            ):
                                                                                qfactor_omega = (
                                                                                    1
                                                                                    / 2
                                                                                    * (
                                                                                        q1prime
                                                                                        - q3prime
                                                                                    )[0]
                                                                                )
                                                                            else:
                                                                                qfactor_omega = (
                                                                                    -1
                                                                                    * (
                                                                                        q1prime
                                                                                        - q3prime
                                                                                    )[
                                                                                        omega
                                                                                    ]
                                                                                )
                                                                            Fprime = (
                                                                                1j
                                                                                * f(
                                                                                    a,
                                                                                    d,
                                                                                    e,
                                                                                )
                                                                                * qfactor_sigma
                                                                                * g_lower_indices[
                                                                                    delta,
                                                                                    omega,
                                                                                ]
                                                                                * qfactor_delta
                                                                                * g_lower_indices[
                                                                                    omega,
                                                                                    sigma,
                                                                                ]
                                                                                * qfactor_omega
                                                                                * g_lower_indices[
                                                                                    sigma,
                                                                                    delta,
                                                                                ]
                                                                            )
                                                                            for mu in [
                                                                                0,
                                                                                1,
                                                                                2,
                                                                                3,
                                                                            ]:
                                                                                for (
                                                                                    nu
                                                                                ) in [
                                                                                    0,
                                                                                    1,
                                                                                    2,
                                                                                    3,
                                                                                ]:
                                                                                    coeff += (
                                                                                        -(
                                                                                            g
                                                                                            ** 2
                                                                                        )
                                                                                        / 2
                                                                                        * (
                                                                                            (
                                                                                                rgpep_factor_first_order(
                                                                                                    Q
                                                                                                    + Qp,
                                                                                                    s,
                                                                                                )
                                                                                            )
                                                                                            * 1
                                                                                            / 2
                                                                                            * (
                                                                                                1
                                                                                                / Q
                                                                                                - 1
                                                                                                / Qp
                                                                                            )
                                                                                            * (
                                                                                                (
                                                                                                    (
                                                                                                        q1
                                                                                                        + q2
                                                                                                    ).dot(
                                                                                                        q1
                                                                                                        + q2
                                                                                                    )
                                                                                                    + (
                                                                                                        q1prime
                                                                                                        + q2prime
                                                                                                    ).dot(
                                                                                                        q1prime
                                                                                                        + q2prime
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                            / (
                                                                                                2
                                                                                                * (
                                                                                                    q3[
                                                                                                        0
                                                                                                    ]
                                                                                                    ** 2
                                                                                                    - 1
                                                                                                    / (
                                                                                                        4
                                                                                                        * K
                                                                                                        ** 2
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                            * g_lower_indices[
                                                                                                0,
                                                                                                0,
                                                                                            ]
                                                                                            * (
                                                                                                g_upper_indices[
                                                                                                    mu,
                                                                                                    gamma,
                                                                                                ]
                                                                                                * F
                                                                                                * (
                                                                                                    A_1[
                                                                                                        alpha
                                                                                                    ]
                                                                                                    * A_2[
                                                                                                        beta
                                                                                                    ]
                                                                                                )
                                                                                                * g_upper_indices[
                                                                                                    nu,
                                                                                                    sigma,
                                                                                                ]
                                                                                                * Fprime
                                                                                                * A_1prime[
                                                                                                    delta
                                                                                                ]
                                                                                                * A_2prime[
                                                                                                    omega
                                                                                                ]
                                                                                            )
                                                                                        )
                                                                                    )

                                                coeff *= (
                                                    1
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

from openparticle.full_dlcq import *
from color_algebra import T, f
from gluon import *
from rgpep import *
import numpy as np
import numba as nb
from itertools import product


@nb.njit(nb.complex128[:, :](nb.complex128[:], nb.float64))
def gamma_slash_plus_m(q, m):
    qplus, qperp1, qperp2 = q[0], q[1], q[2]
    return (  #
        0.5 * gamma_plus * (m**2 / qplus)
        + 0.5 * gamma_minus * qplus
        - gamma1 * qperp1
        - gamma2 * qperp2
        + m * np.eye(2)
    )


quark_helicities = np.array([1, -1])
gluon_polarizations = np.array([1, -1])


spin_arrays_qqqq = np.array(
    list(
        product(
            quark_helicities,
            quark_helicities,
            quark_helicities,
            quark_helicities,
        )
    ),
    dtype=np.int8,
)
color_combos_qqqq = np.array(
    [
        [0, 0, 2, 0, 0],
        [0, 0, 2, 1, 1],
        [0, 0, 7, 0, 0],
        [0, 0, 7, 1, 1],
        [0, 0, 7, 2, 2],
        [0, 1, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 2, 3, 0, 2],
        [0, 2, 3, 2, 0],
        [0, 2, 4, 0, 2],
        [0, 2, 4, 2, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 1, 0],
        [1, 0, 1, 0, 1],
        [1, 1, 2, 1, 1],
        [1, 1, 7, 1, 1],
        [1, 2, 5, 1, 2],
        [1, 2, 5, 2, 1],
        [1, 2, 6, 1, 2],
        [1, 2, 6, 2, 1],
        [2, 0, 3, 2, 0],
        [2, 0, 3, 0, 2],
        [2, 0, 4, 2, 0],
        [2, 0, 4, 0, 2],
        [2, 1, 5, 2, 1],
        [2, 1, 5, 1, 2],
        [2, 1, 6, 2, 1],
        [2, 1, 6, 1, 2],
        [2, 2, 7, 2, 2],
    ],
    dtype=np.int8,
)  # c1, c2, a, c1', c2'

fixed_qnums_qqqq = np.vstack(
    [np.hstack([row1, spin]) for row1 in color_combos_qqqq for spin in spin_arrays_qqqq]
)

color_combos = np.array(
    [
        [0, 0, 2, 2],
        [0, 0, 7, 2],
        [0, 0, 2, 7],
        [0, 0, 7, 7],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 1],
        [0, 2, 3, 3],
        [0, 2, 4, 3],
        [0, 2, 3, 4],
        [0, 2, 4, 4],
        [1, 0, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 1],
        [1, 1, 2, 2],
        [1, 1, 7, 2],
        [1, 1, 2, 7],
        [1, 1, 7, 7],
        [1, 2, 5, 5],
        [1, 2, 6, 5],
        [1, 2, 5, 6],
        [1, 2, 6, 6],
        [2, 0, 3, 3],
        [2, 0, 4, 3],
        [2, 0, 3, 4],
        [2, 0, 4, 4],
        [2, 1, 5, 5],
        [2, 1, 6, 5],
        [2, 1, 5, 6],
        [2, 1, 6, 6],
        [2, 2, 7, 7],
    ]
)  # a, b, c1, c2: T[a, c1, c2] * T[b, c1, c2]

spin_arrays_qgqg = np.array(
    list(
        product(
            quark_helicities, gluon_polarizations, gluon_polarizations, quark_helicities
        )
    ),
    dtype=np.int8,
)

fixed_qnums_qgqg = np.vstack(
    [np.hstack([row1, spin]) for row1 in color_combos for spin in spin_arrays_qgqg]
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
def effective_quark_legs_term_tensor(
    K: float,
    Kp: float,
    s: float,
    g: float,
    mq: float,
    mg: float,
):
    """
    Indices 0, 3, 6 correspond respectively to q1+, q2+, q3+.
    """

    fermion_lim = K - ((K - 0.5) % 1)
    fermion_longitudinal_q = np.array(
        [i for i in np.arange(-fermion_lim, fermion_lim + 1.0, 1.0) if i != 0.0],
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
        len(fermion_longitudinal_q),
        len(transverse_q),
        len(transverse_q),
        len(fixed_qnums_qqqq),
    )
    gluon_exch_tensor = np.zeros(dim, dtype=np.complex128)

    for idx in nb.prange(len(fermion_longitudinal_q)):
        q1plus = fermion_longitudinal_q[idx]
        for q1perp1 in transverse_q:
            for q1perp2 in transverse_q:
                for q2plus in fermion_longitudinal_q:
                    for q2perp1 in transverse_q:
                        for q2perp2 in transverse_q:
                            for q1primeplus in fermion_longitudinal_q:
                                for q1primeperp1 in transverse_q:
                                    for q1primeperp2 in transverse_q:
                                        for (
                                            fixed_set_qnums_index,
                                            fixed_set_qnums,
                                        ) in enumerate(fixed_qnums_qqqq):
                                            a, c1, c2 = (
                                                fixed_set_qnums[2] + 1,
                                                fixed_set_qnums[0],
                                                fixed_set_qnums[1],
                                            )
                                            c3, c4 = (
                                                fixed_set_qnums[3],
                                                fixed_set_qnums[4],
                                            )
                                            (
                                                helicity1,
                                                helicity2,
                                                helicity1prime,
                                                helicity2prime,
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
                                            q3 = -1 * (q1 + q2)
                                            q2prime = -1 * (q1prime - q3)

                                            if (
                                                0 < np.abs(q3[0]) <= K
                                                and np.abs(q3[1]) <= Kp
                                                and np.abs(q3[2]) <= Kp
                                                and 0 < np.abs(q2prime[0]) <= K
                                                and np.abs(q2prime[1]) <= Kp
                                                and np.abs(q2prime[2]) <= Kp
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
                                                psi_1prime = heaviside(
                                                    -q1prime[0]
                                                ) * udag(
                                                    p=-q1prime, m=mq, h=helicity1prime
                                                ) + heaviside(
                                                    q1prime[0]
                                                ) * vdag(
                                                    p=q1prime, m=mq, h=helicity1prime
                                                )
                                                psi_2prime = heaviside(q2prime[0]) * u(
                                                    p=q2prime, m=mq, h=helicity2prime
                                                ) + heaviside(-q2prime[0]) * v(
                                                    p=-q2prime, m=mq, h=helicity2prime
                                                )
                                                coeff = 0
                                                for mu in [0, 1, 2, 3]:
                                                    for nu in [0, 1, 2, 3]:
                                                        coeff += psi_1.dot(
                                                            gamma0.dot(
                                                                gamma[mu].dot(psi_2)
                                                            )
                                                            * d(q3, mg, mu, nu, "lower")
                                                            * psi_1prime.dot(
                                                                gamma[nu].dot(
                                                                    psi_2prime
                                                                )
                                                            )
                                                        )
                                                q3prime = -q3  # delta(q3 + q3prime)
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
                                                    * T[a, c3, c4]
                                                    * 1
                                                    / (
                                                        np.abs(q1[0])
                                                        * np.abs(q2[0])
                                                        * np.abs(q3[0])
                                                        * np.abs(q1prime[0])
                                                        * np.abs(q2prime[0])
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


@nb.njit(
    ## output types
    nb.complex128[:, :, :, :, :, :, :, :, :, :]
    ##input types
    (nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64),
    ## other options
    fastmath=True,
    parallel=True,
)
def effective_quark_exchange_mixed_legs_term_tensor(
    K: float,
    Kp: float,
    s: float,
    g: float,
    mq: float,
    mg: float,
):
    boson_lim = int(K)
    fermion_lim = K - ((K - 0.5) % 1)
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
        len(boson_longitudinal_q),
        len(transverse_q),
        len(transverse_q),
        # particle 3
        len(boson_longitudinal_q),
        len(transverse_q),
        len(transverse_q),
        len(fixed_qnums_qgqg),
    )

    quark_exch_tensor = np.zeros(dim, dtype=np.complex128)
    for idx in nb.prange(len(fermion_longitudinal_q)):
        q1plus = fermion_longitudinal_q[idx]
        for q1perp1 in transverse_q:
            for q1perp2 in transverse_q:
                for q3plus in boson_longitudinal_q:
                    for q3perp1 in transverse_q:
                        for q3perp2 in transverse_q:
                            for q3primeplus in boson_longitudinal_q:
                                for q3primeperp1 in transverse_q:
                                    for q3primeperp2 in transverse_q:
                                        for (
                                            fixed_set_qnums_index,
                                            fixed_set_qnums,
                                        ) in enumerate(fixed_qnums_qgqg):
                                            c1, c1prime, a, b = (
                                                fixed_set_qnums[0],
                                                fixed_set_qnums[1],
                                                fixed_set_qnums[2],
                                                fixed_set_qnums[3],
                                            )
                                            (
                                                helicity1,
                                                helicity1prime,
                                                polarization3,
                                                polarization3prime,
                                            ) = (
                                                fixed_set_qnums[4],
                                                fixed_set_qnums[7],
                                                fixed_set_qnums[5],
                                                fixed_set_qnums[6],
                                            )
                                            q1 = np.array(
                                                [
                                                    q1plus,
                                                    q1perp1,
                                                    q1perp2,
                                                ],
                                                dtype=np.complex128,
                                            )
                                            q3 = np.array(
                                                [
                                                    q3plus,
                                                    q3perp1,
                                                    q3perp2,
                                                ],
                                                dtype=np.complex128,
                                            )
                                            q3prime = np.array(
                                                [
                                                    q3primeplus,
                                                    q3primeperp1,
                                                    q3primeperp2,
                                                ],
                                                dtype=np.complex128,
                                            )
                                            q2 = -1 * (
                                                q1 + q3
                                            )  # delta(q1 + q2 + q3) from first order vertex
                                            q2prime = (
                                                -q2
                                            )  # delta(q2 + q2prime) from contraction
                                            q1prime = -1 * (
                                                q2prime + q3prime
                                            )  # delta(q1' + q2' + q3') from first order vertex

                                            if (
                                                0 < np.abs(q2[0]) <= K
                                                and np.abs(q2[1]) <= Kp
                                                and np.abs(q2[2]) <= Kp
                                                and 0 < np.abs(q2prime[0]) <= K
                                                and np.abs(q2prime[1]) <= Kp
                                                and np.abs(q2prime[2]) <= Kp
                                                and 0 < np.abs(q1prime[0]) <= K
                                                and np.abs(q1prime[1]) <= Kp
                                                and np.abs(q1prime[2]) <= Kp
                                            ):

                                                psi1 = heaviside(-q1[0]) * udag(
                                                    p=-q1, m=mq, h=helicity1
                                                ) + heaviside(q1[0]) * (
                                                    vdag(p=q1, m=mq, h=helicity1)
                                                )

                                                A3 = (
                                                    heaviside(q3[0])
                                                    * pol_vec(p=q3, pol=polarization3)
                                                    + heaviside(-q3[0])
                                                    * pol_vec(
                                                        p=-q3, pol=polarization3
                                                    ).conj()
                                                )

                                                A3prime = (
                                                    heaviside(q3prime[0])
                                                    * pol_vec(
                                                        p=q3prime, pol=polarization3
                                                    )
                                                    + heaviside(-q3prime[0])
                                                    * pol_vec(
                                                        p=-q3prime, pol=polarization3
                                                    ).conj()
                                                )

                                                A3slash = (
                                                    0.5 * gamma_plus * (A3[1])
                                                    - gamma1 * (A3[2])
                                                    - gamma2 * (A3[3])
                                                )
                                                A3primeslash = (
                                                    0.5 * gamma_plus * (A3prime[1])
                                                    - gamma1 * (A3prime[2])
                                                    - gamma2 * (A3prime[3])
                                                )

                                                psi1prime = heaviside(q1prime[0]) * u(
                                                    p=q1prime, m=mq, h=helicity1prime
                                                ) + heaviside(-q1prime[0]) * v(
                                                    p=-q1prime, m=mq, h=helicity1prime
                                                )

                                                coeff = psi1.dot(
                                                    gamma0.dot(
                                                        A3slash.dot(
                                                            gamma_slash_plus_m(
                                                                q2, mq
                                                            ).dot(
                                                                A3primeslash.dot(
                                                                    psi1prime
                                                                )
                                                            )
                                                        )
                                                    )
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
                                                    * T[a][c1][c1prime]
                                                    * T[b][c1][c1prime]
                                                    * 1
                                                    / (q2[0])
                                                    * 1
                                                    / (
                                                        np.abs(q1[0])
                                                        * np.abs(q3[0])
                                                        * np.abs(q1prime[0])
                                                        * np.abs(q3prime[0])
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

                                                    q3plus_index = np.where(
                                                        boson_longitudinal_q == q3plus
                                                    )[0][0]
                                                    q3perp1_index = np.where(
                                                        transverse_q == q3perp1
                                                    )[0][0]
                                                    q3perp2_index = np.where(
                                                        transverse_q == q3perp2
                                                    )[0][0]

                                                    q3primeplus_index = np.where(
                                                        boson_longitudinal_q
                                                        == q3prime[0]
                                                    )[0][0]
                                                    q3primeperp1_index = np.where(
                                                        transverse_q == q3prime[1]
                                                    )[0][0]
                                                    q3primeperp2_index = np.where(
                                                        transverse_q == q3prime[2]
                                                    )[0][0]

                                                    quark_exch_tensor[
                                                        q1plus_index,
                                                        q1perp1_index,
                                                        q1perp2_index,
                                                        q3plus_index,
                                                        q3perp1_index,
                                                        q3perp2_index,
                                                        q3primeplus_index,
                                                        q3primeperp1_index,
                                                        q3primeperp2_index,
                                                        fixed_set_qnums_index,
                                                    ] = coeff

    return quark_exch_tensor

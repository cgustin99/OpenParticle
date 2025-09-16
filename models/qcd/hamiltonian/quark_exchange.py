from openparticle.full_dlcq import *
from color_algebra import T, f
import numpy as np
import numba as nb
from itertools import product

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
)
quark_helicities = np.array([1, -1])
gluon_polarizations = np.array([1, -1])
spin_arrays = np.array(
    list(
        product(
            quark_helicities, gluon_polarizations, gluon_polarizations, quark_helicities
        )
    ),
    dtype=np.int8,
)

fixed_qnums = np.vstack(
    [np.hstack([row1, spin]) for row1 in color_combos for spin in spin_arrays]
)


@nb.njit(
    nb.complex128[:, :, :, :, :, :, :, :, :, :](
        nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64
    ),
    fastmath=True,
    parallel=True,
)
def quark_exchange_mixed_legs_term_tensor(
    s: float, K: float, Kp: float, g: float, mq: float, mg: float
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
        len(fixed_qnums),
    )

    quark_exch_tensor = np.zeros(dim, dtype=np.complex128)
    for idx in nb.prange(len(fermion_longitudinal_q)):
        q1plus = fermion_longitudinal_q[idx]
        for q1perp1 in transverse_q:
            for q1perp2 in transverse_q:
                for q2plus in boson_longitudinal_q:
                    for q2perp1 in transverse_q:
                        for q2perp2 in transverse_q:
                            for q3plus in boson_longitudinal_q:
                                for q3perp1 in transverse_q:
                                    for q3perp2 in transverse_q:
                                        for (
                                            fixed_set_qnums_index,
                                            fixed_set_qnums,
                                        ) in enumerate(fixed_qnums):
                                            c1, c4, a = (
                                                fixed_set_qnums[0],
                                                fixed_set_qnums[1],
                                                fixed_set_qnums[2],
                                            )
                                            b = fixed_set_qnums[3]
                                            (
                                                helicity1,
                                                helicity4,
                                                polarization2,
                                                polarization3,
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
                                            q2 = np.array(
                                                [
                                                    q2plus,
                                                    q2perp1,
                                                    q2perp2,
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
                                            q4 = -1 * (
                                                q3 + q2 + q1
                                            )  # q4 is assigned by delta

                                            if (
                                                0 < np.abs(q4[0]) <= K
                                                and np.abs(q4[1]) <= Kp
                                                and np.abs(q4[2]) <= Kp
                                            ):  # make sure k4 < K,Kp
                                                q1minus = (
                                                    mq**2 + q1[1:3].dot(q1[1:3])
                                                ) / q1[0]
                                                q2minus = (
                                                    mg**2 + q2[1:3].dot(q2[1:3])
                                                ) / q2[0]
                                                q3minus = (
                                                    mg**2 + q3[1:3].dot(q3[1:3])
                                                ) / q3[0]
                                                q4minus = (
                                                    mq**2 + q4[1:3].dot(q4[1:3])
                                                ) / q4[0]

                                                psi1 = heaviside(-q1[0]) * udag(
                                                    p=-q1, m=mq, h=helicity1
                                                ) + heaviside(q1[0]) * (
                                                    vdag(p=q1, m=mq, h=helicity1)
                                                )

                                                A2 = (
                                                    heaviside(q2[0])
                                                    * pol_vec(p=q2, pol=polarization2)
                                                    + heaviside(-q2[0])
                                                    * pol_vec(
                                                        p=-q2, pol=polarization2
                                                    ).conj()
                                                )

                                                A3 = (
                                                    heaviside(q3[0])
                                                    * pol_vec(p=q3, pol=polarization3)
                                                    + heaviside(-q3[0])
                                                    * pol_vec(
                                                        p=-q3, pol=polarization3
                                                    ).conj()
                                                )

                                                A2slash = (
                                                    0.5 * gamma_plus * (A2[1])
                                                    - gamma1 * (A2[2])
                                                    - gamma2 * (A2[3])
                                                )
                                                A3slash = (
                                                    0.5 * gamma_plus * (A3[1])
                                                    - gamma1 * (A3[2])
                                                    - gamma2 * (A3[3])
                                                )

                                                psi4 = heaviside(q4[0]) * u(
                                                    p=q4, m=mq, h=helicity4
                                                ) + heaviside(-q4[0]) * v(
                                                    p=-q4, m=mq, h=helicity4
                                                )

                                                coeff = psi1.dot(
                                                    gamma0.dot(
                                                        A2slash.dot(
                                                            gamma_plus.dot(
                                                                A3slash.dot(psi4)
                                                            )
                                                        )
                                                    )
                                                )

                                                coeff *= (
                                                    g**2
                                                    * T[a][c1][c4]
                                                    * T[b][c1][c4]
                                                    * 1
                                                    / (q3[0] + q4[0])
                                                    * 1
                                                    / (
                                                        np.abs(q1[0])
                                                        * np.abs(q2[0])
                                                        * np.abs(q3[0])
                                                        * np.abs(q4[0])
                                                    )
                                                    * np.exp(
                                                        -s
                                                        * (
                                                            q1minus
                                                            + q2minus
                                                            + q3minus
                                                            + q4minus
                                                        )
                                                        ** 2
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

                                                    quark_exch_tensor[
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

    return quark_exch_tensor

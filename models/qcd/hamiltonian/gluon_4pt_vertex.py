from openparticle.full_dlcq import *
from color_algebra import T, f
import numpy as np
import numba as nb
from itertools import product

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
    ]
)  # a, b, c, d, e
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
def gluon_4pt_vertex_term_tensor(s: float, K: float, Kp: float, g: float, mg: float):
    """
    Indices 0, 3, 6 correspond respectively to q1+, q2+, q3+.
    """

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
    gluon_4pt_vertex_tensor = np.zeros(dim, dtype=np.complex128)

    for idx in nb.prange(len(boson_longitudinal_q)):
        q1plus = boson_longitudinal_q[idx]
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
                                            a, b, c = (
                                                fixed_set_qnums[0] - 1,
                                                fixed_set_qnums[1] - 1,
                                                fixed_set_qnums[2] - 1,
                                            )
                                            d, e = (
                                                fixed_set_qnums[3] - 1,
                                                fixed_set_qnums[4] - 1,
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
                                                    mg**2 + q1[1:3].dot(q1[1:3])
                                                ) / q1[0]
                                                q2minus = (
                                                    mg**2 + q2[1:3].dot(q2[1:3])
                                                ) / q2[0]
                                                q3minus = (
                                                    mg**2 + q3[1:3].dot(q3[1:3])
                                                ) / q3[0]
                                                q4minus = (
                                                    mg**2 + q4[1:3].dot(q4[1:3])
                                                ) / q4[0]
                                                Aq1 = (
                                                    heaviside(q1[0])
                                                    * pol_vec(
                                                        p=q1,
                                                        pol=polarization1,
                                                    )[2:4]
                                                    + heaviside(-q1[0])
                                                    * pol_vec(
                                                        p=-q1,
                                                        pol=polarization1,
                                                    ).conj()[2:4]
                                                )
                                                Aq3 = (
                                                    heaviside(q3[0])
                                                    * pol_vec(
                                                        p=q3,
                                                        pol=polarization3,
                                                    )[2:4]
                                                    + heaviside(-q3[0])
                                                    * pol_vec(
                                                        p=-q3,
                                                        pol=polarization3,
                                                    ).conj()[2:4]
                                                )
                                                Aq2 = (
                                                    heaviside(q2[0])
                                                    * pol_vec(
                                                        p=q2,
                                                        pol=polarization2,
                                                    )[2:4]
                                                    + heaviside(-q2[0])
                                                    * pol_vec(
                                                        p=-q2,
                                                        pol=polarization2,
                                                    ).conj()[2:4]
                                                )
                                                Aq4 = (
                                                    heaviside(q4[0])
                                                    * pol_vec(
                                                        p=q4,
                                                        pol=polarization4,
                                                    )[2:4]
                                                    + heaviside(-q4[0])
                                                    * pol_vec(
                                                        p=-q4,
                                                        pol=polarization4,
                                                    ).conj()[2:4]
                                                )

                                                coeff = Aq1.dot(Aq3) * Aq2.dot(Aq4)
                                                coeff *= (
                                                    g**2
                                                    * f(
                                                        a,
                                                        b,
                                                        c,
                                                    )
                                                    * f(
                                                        a,
                                                        d,
                                                        e,
                                                    )
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

                                                    gluon_4pt_vertex_tensor[
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

    return gluon_4pt_vertex_tensor

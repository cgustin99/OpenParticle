from openparticle.full_dlcq import *
from color_algebra import T, f
import numpy as np
import numba as nb
from itertools import product

color_combos = np.array(
    [
        [0, 1, 2],
        [0, 3, 6],
        [0, 4, 5],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 4],
        [2, 5, 6],
        [3, 4, 7],
        [5, 6, 7],
    ]
)
quark_helicities = np.array([1, -1])
gluon_polarizations = np.array([1, -1])
spin_arrays = np.array(
    list(product(quark_helicities, quark_helicities, gluon_polarizations)),
    dtype=np.int8,
)

fixed_qnums = np.vstack(
    [np.hstack([row1, spin]) for row1 in color_combos for spin in spin_arrays]
)


@nb.njit(
    ## output types
    nb.complex128[:, :, :, :, :, :, :]
    ##input types
    (nb.float64, nb.float64, nb.float64, nb.int8),
    ## other options
    fastmath=True,
)
def gluon_3pt_vertex_term_tensor(K: float, Kp: float, g: float, Nc: int):
    """
    Indices 0, 3 correspond respectively to q1+, q2+.
    """

    boson_lim = int(K)
    boson_longitudinal_q = np.arange(1, boson_lim + 1, 1, dtype=np.float64)

    transverse_q = np.array([i for i in np.arange(-Kp, Kp + 1.0)], dtype=np.float64)

    dim = (
        # particle 1 momentum
        len(boson_longitudinal_q),
        len(transverse_q),
        len(transverse_q),
        # particle 2 momentum
        len(boson_longitudinal_q),
        len(transverse_q),
        len(transverse_q),
        # fixed quantum numbers array
        len(fixed_qnums),
    )
    gluon_3pt_vertex_tensor = np.zeros(dim, dtype=np.complex128)

    for idx in nb.prange(len(boson_longitudinal_q)):
        q1plus = boson_longitudinal_q[idx]
        for q1perp1 in transverse_q:
            for q1perp2 in transverse_q:
                for q2plus in boson_longitudinal_q:
                    for q2perp1 in transverse_q:
                        for q2perp2 in transverse_q:
                            for fixed_set_qnums_index, fixed_set_qnums in enumerate(
                                fixed_qnums
                            ):
                                color1, color2, color3 = (
                                    fixed_set_qnums[0],
                                    fixed_set_qnums[1],
                                    fixed_set_qnums[2],
                                )
                                polarization1, polarization2, polarization3 = (
                                    fixed_set_qnums[3],
                                    fixed_set_qnums[4],
                                    fixed_set_qnums[5],
                                )
                                q1 = np.array(
                                    [q1plus, q1perp1, q1perp2],
                                    dtype=np.complex128,
                                )
                                q2 = np.array(
                                    [q2plus, q2perp1, q2perp2],
                                    dtype=np.complex128,
                                )
                                q3 = -1 * (q2 + q1)  # q3 is assigned by delta
                                if (
                                    0 < np.abs(q3[0]) <= K
                                    and np.abs(q3[1]) <= Kp
                                    and np.abs(q3[2]) <= Kp
                                ):  # make sure k3 < K,Kp
                                    Aone = (
                                        heaviside(q1[0])
                                        * pol_vec(p=q1, pol=polarization1)
                                        + heaviside(-q1[0])
                                        * pol_vec(p=q1, pol=polarization1).conj()
                                    )

                                    Atwo = (
                                        heaviside(q2[0])
                                        * pol_vec(p=q2, pol=polarization2)
                                        + heaviside(-q2[0])
                                        * pol_vec(p=q2, pol=polarization2).conj()
                                    )

                                    Athree = (
                                        heaviside(q3[0])
                                        * pol_vec(p=q3, pol=polarization3)
                                        + heaviside(-q3[0])
                                        * pol_vec(p=q3, pol=polarization3).conj()
                                    )

                                    coeff = (
                                        q2[0]
                                        * (Aone[2] * Atwo[2] + Aone[3] * Atwo[3])
                                        * Athree[1]
                                        - q2[1]
                                        * (Aone[2] * Atwo[2] + Aone[3] * Atwo[3])
                                        * Athree[2]
                                        - q2[2]
                                        * (Aone[2] * Atwo[2] + Aone[3] * Atwo[3])
                                        * Athree[3]
                                    )
                                    coeff *= (
                                        -1j
                                        * g
                                        * f(color1, color2, color3)
                                        * 1
                                        / (
                                            np.abs(q1[0])
                                            * np.abs(q2[0])
                                            * np.abs(q3[0])
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

                                        gluon_3pt_vertex_tensor[
                                            q1plus_index,
                                            q1perp1_index,
                                            q1perp2_index,
                                            q2plus_index,
                                            q2perp1_index,
                                            q2perp2_index,
                                            fixed_set_qnums_index,
                                        ] = coeff

    return gluon_3pt_vertex_tensor

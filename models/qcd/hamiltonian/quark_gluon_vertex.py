import openparticle as op
from openparticle.full_dlcq import *
from color_algebra import T
import numpy as np
import numba as nb


@nb.njit(
    ## output types
    nb.complex128[:, :, :, :, :, :, :, :, :, :, :, :, :, :, :]
    ##input types
    (nb.float64, nb.float64, nb.float64, nb.float64, nb.int8),
    ## other options
    fastmath=True,
)
def quark_gluon_vertex_term_tensor(
    K: float, Kp: float, g: float, mq: float, Nc: int = 3
):
    """
    Indices 2, 7, 12 correspond respectively to q1+, q2+, q3+
    """

    quark_helicities = [1, -1]
    quark_colors = np.arange(1, Nc + 1, 1)
    gluon_polarizations = [1, -1]
    gluon_colors = np.arange(1, Nc**2 - 1 + 1, 1)

    fermion_lim = K - ((K - 0.5) % 1)
    boson_lim = int(K)
    boson_longitudinal_q = np.arange(1, boson_lim + 1, 1, dtype=np.float64)
    fermion_longitudinal_q = np.array(
        [i for i in np.arange(-fermion_lim, fermion_lim + 1.0, 1.0) if i != 0.0],
        dtype=np.float64,
    )
    transverse_q = np.array([i for i in np.arange(-Kp, Kp + 1.0)], dtype=np.float64)

    dim = (  # particle 1
        len(quark_colors),
        len(quark_helicities),
        len(fermion_longitudinal_q),
        len(transverse_q),
        len(transverse_q),
        # particle 2
        len(quark_colors),
        len(quark_helicities),
        len(fermion_longitudinal_q),
        len(transverse_q),
        len(transverse_q),
        # particle 3
        len(gluon_colors),
        len(gluon_polarizations),
        len(boson_longitudinal_q),
        len(transverse_q),
        len(transverse_q),
    )
    quark_gluon_vertex_tensor = np.zeros(dim, dtype=np.complex128)

    for color1 in quark_colors:
        for helicity1 in quark_helicities:
            for q1plus in fermion_longitudinal_q:
                for q1perp1 in transverse_q:
                    for q1perp2 in transverse_q:
                        for color2 in quark_colors:
                            for helicity2 in quark_helicities:
                                for q2plus in fermion_longitudinal_q:
                                    for q2perp1 in transverse_q:
                                        for q2perp2 in transverse_q:
                                            for gluon_color in gluon_colors:
                                                for (
                                                    gluon_polarization
                                                ) in gluon_polarizations:
                                                    q1 = np.array(
                                                        [q1plus, q1perp1, q1perp2],
                                                        dtype=np.complex128,
                                                    )
                                                    q2 = np.array(
                                                        [q2plus, q2perp1, q2perp2],
                                                        dtype=np.complex128,
                                                    )
                                                    q3 = -1 * (
                                                        q2 + q1
                                                    )  # q3 is assigned by delta

                                                    if (
                                                        0 < np.abs(q3[0]) <= K
                                                        and np.abs(q3[1]) <= Kp
                                                        and np.abs(q3[2]) <= Kp
                                                    ):  # make sure k3 < K,Kp
                                                        e = pol_vec(
                                                            p=q3, pol=gluon_polarization
                                                        )
                                                        estar = pol_vec(
                                                            p=-q3,
                                                            pol=gluon_polarization,
                                                        ).conj()

                                                        e_slash = (
                                                            0.5 * gamma_plus * (e[1])
                                                            - gamma1 * (e[2])
                                                            - gamma2 * (e[3])
                                                        )
                                                        estar_slash = (
                                                            0.5
                                                            * gamma_plus
                                                            * (estar[1])
                                                            - gamma1 * (estar[2])
                                                            - gamma2 * (estar[3])
                                                        )

                                                        psi_dag = heaviside(
                                                            -q1[0]
                                                        ) * udag(
                                                            p=-q1, m=mq, h=helicity1
                                                        ) + heaviside(
                                                            q1[0]
                                                        ) * vdag(
                                                            p=q1, m=mq, h=helicity1
                                                        )
                                                        A = (
                                                            heaviside(q3[0]) * e_slash
                                                            + heaviside(-q3[0])
                                                            * estar_slash
                                                        )
                                                        psi = heaviside(q2[0]) * u(
                                                            p=q2, m=mq, h=helicity2
                                                        ) + heaviside(-q2[0]) * v(
                                                            p=-q2, m=mq, h=helicity2
                                                        )

                                                        coeff = (
                                                            g
                                                            * T[gluon_color - 1][
                                                                color1 - 1
                                                            ][color2 - 1]
                                                            * (
                                                                psi_dag.dot(
                                                                    gamma0.dot(
                                                                        A.dot(psi)
                                                                    )
                                                                )
                                                            )
                                                        )
                                                        if coeff != 0:
                                                            color1_index = color1 - 1
                                                            helicity1_index = (
                                                                helicity1 - 1
                                                            )
                                                            q1plus_index = np.where(
                                                                boson_longitudinal_q
                                                                == q1plus
                                                            )[0][0]
                                                            q1perp1_index = np.where(
                                                                transverse_q == q1perp1
                                                            )[0][0]
                                                            q1perp2_index = np.where(
                                                                transverse_q == q1perp2
                                                            )[0][0]

                                                            color2_index = color2 - 1
                                                            helicity2_index = (
                                                                helicity2 - 1
                                                            )
                                                            q2plus_index = np.where(
                                                                boson_longitudinal_q
                                                                == q2plus
                                                            )[0][0]
                                                            q2perp1_index = np.where(
                                                                transverse_q == q2perp1
                                                            )[0][0]
                                                            q2perp2_index = np.where(
                                                                transverse_q == q2perp2
                                                            )[0][0]

                                                            gluon_color_index = (
                                                                gluon_color - 1
                                                            )
                                                            gluon_polarization_index = (
                                                                gluon_polarization - 1
                                                            )
                                                            q3plus_index = np.where(
                                                                boson_longitudinal_q
                                                                == q3[0]
                                                            )[0][0]
                                                            q3perp1_index = np.where(
                                                                transverse_q == q3[1]
                                                            )[0][0]
                                                            q3perp2_index = np.where(
                                                                transverse_q == q3[2]
                                                            )[0][0]

                                                            quark_gluon_vertex_tensor[
                                                                color1_index,
                                                                helicity1_index,
                                                                q1plus_index,
                                                                q1perp1_index,
                                                                q1perp2_index,
                                                                color2_index,
                                                                helicity2_index,
                                                                q2plus_index,
                                                                q2perp1_index,
                                                                q2perp2_index,
                                                                gluon_color_index,
                                                                gluon_polarization_index,
                                                                q3plus_index,
                                                                q3perp1_index,
                                                                q3perp2_index,
                                                            ] = coeff

    return quark_gluon_vertex_tensor

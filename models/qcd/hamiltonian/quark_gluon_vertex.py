from openparticle.full_dlcq import *
from color_algebra import T
import numpy as np
import numba as nb
from itertools import product

color_combos = np.array(
    [
        [0, 0, 2],
        [0, 0, 7],
        [0, 1, 0],
        [0, 1, 1],
        [0, 2, 3],
        [0, 2, 4],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 2],
        [1, 1, 7],
        [1, 2, 5],
        [1, 2, 6],
        [2, 0, 3],
        [2, 0, 4],
        [2, 1, 5],
        [2, 1, 6],
        [2, 2, 7],
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
    (nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64),
    ## other options
    fastmath=True,
    parallel=True,
)
def quark_gluon_vertex_term_tensor(
    s: float, K: float, Kp: float, g: float, mq: float, mg: float
):
    """
    Indices 0, 3 correspond respectively to q1+, q2+.
    """

    fermion_lim = K - ((K - 0.5) % 1)
    fermion_longitudinal_q = np.array(
        [i for i in np.arange(-fermion_lim, fermion_lim + 1.0, 1.0) if i != 0.0],
        dtype=np.float64,
    )
    boson_lim = int(K)
    boson_longitudinal_q = np.arange(-boson_lim, boson_lim + 1, 1, dtype=np.float64)
    transverse_q = np.array([i for i in np.arange(-Kp, Kp + 1.0)], dtype=np.float64)
    dim = (
        # particle 1 momentum
        len(fermion_longitudinal_q),
        len(transverse_q),
        len(transverse_q),
        # particle 2 momentum
        len(fermion_longitudinal_q),
        len(transverse_q),
        len(transverse_q),
        # fixed quantum numbers array
        len(fixed_qnums),
    )

    quark_gluon_vertex_tensor = np.zeros(dim, dtype=np.complex128)

    for idx in nb.prange(len(fermion_longitudinal_q)):
        q1plus = fermion_longitudinal_q[idx]
        for q1perp1 in transverse_q:
            for q1perp2 in transverse_q:
                for q2plus in fermion_longitudinal_q:
                    for q2perp1 in transverse_q:
                        for q2perp2 in transverse_q:
                            for fixed_set_qnums_index, fixed_set_qnums in enumerate(
                                fixed_qnums
                            ):

                                color1, color2, gluon_color = (
                                    fixed_set_qnums[0],
                                    fixed_set_qnums[1],
                                    fixed_set_qnums[2],
                                )
                                helicity1, helicity2, gluon_polarization = (
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

                                q1minus = (mq**2 + q1[1:3].dot(q1[1:3])) / q1[0]
                                q2minus = (mq**2 + q2[1:3].dot(q2[1:3])) / q2[0]
                                q3minus = (mg**2 + q3[1:3].dot(q3[1:3])) / q3[0]

                                if (
                                    0 < np.abs(q3[0]) <= K
                                    and np.abs(q3[1]) <= Kp
                                    and np.abs(q3[2]) <= Kp
                                ):  # make sure k3 < K,Kp
                                    e = pol_vec(p=q3, pol=gluon_polarization)
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
                                        0.5 * gamma_plus * (estar[1])
                                        - gamma1 * (estar[2])
                                        - gamma2 * (estar[3])
                                    )

                                    psi_dag = heaviside(-q1[0]) * udag(
                                        p=-q1, m=mq, h=helicity1
                                    ) + heaviside(q1[0]) * vdag(p=q1, m=mq, h=helicity1)
                                    A = (
                                        heaviside(q3[0]) * e_slash
                                        + heaviside(-q3[0]) * estar_slash
                                    )
                                    psi = heaviside(q2[0]) * u(
                                        p=q2, m=mq, h=helicity2
                                    ) + heaviside(-q2[0]) * v(p=-q2, m=mq, h=helicity2)

                                    coeff = (
                                        g
                                        * T[gluon_color][color1][color2]
                                        * (psi_dag.dot(gamma0.dot(A.dot(psi))))
                                        * np.exp(
                                            -s * (q1minus + q2minus + q3minus) ** 2
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

                                        quark_gluon_vertex_tensor[
                                            q1plus_index,
                                            q1perp1_index,
                                            q1perp2_index,
                                            q2plus_index,
                                            q2perp1_index,
                                            q2perp2_index,
                                            fixed_set_qnums_index,
                                        ] = coeff

    return quark_gluon_vertex_tensor

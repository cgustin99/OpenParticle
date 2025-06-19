import openparticle as op
from openparticle.full_dlcq import *
from color_algebra import T, f
import numpy as np
import numba as nb


@nb.njit(
    ## output types
    nb.types.DictType(nb.types.string, nb.types.complex128)
    ##input types
    (
        nb.int64,
        nb.int64,
        nb.int64,
        nb.int64,
        nb.int64,
        nb.int64,
        nb.int64,
        nb.int64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.int64,
    ),
    ## other options
    fastmath=True,
)
def fixed_fermion_exch_inst_term(
    lambda_1: int,
    sigma_2: int,
    sigma_3: int,
    lambda_4: int,
    c1: int,
    c4: int,
    a: int,
    b: int,
    g: float,
    mq: float,
    K: float,
    Kp: float,
    Nc: int = 3,
):

    fixed_qns_dict = nb.typed.Dict.empty(
        key_type=nb.types.unicode_type, value_type=nb.types.complex128
    )
    boson_lim = int(K)
    fermion_lim = K - ((K - 0.5) % 1)

    L = 2 * np.pi * K
    Lp = 2 * np.pi * Kp

    fermion_longitudinal_q = np.array(
        [i for i in np.arange(-fermion_lim, fermion_lim + 1.0, 1.0) if i != 0.0],
        dtype=np.float64,
    )
    boson_longitudinal_q = np.array(
        [i for i in np.arange(-boson_lim, boson_lim + 1.0, 1.0) if i != 0.0],
        dtype=np.float64,
    )
    transverse_q = np.array([i for i in np.arange(-Kp, Kp + 1.0)], dtype=np.float64)

    for q1plus in fermion_longitudinal_q:
        for q1trans1 in transverse_q:
            for q1trans2 in transverse_q:
                for q2plus in boson_longitudinal_q:
                    for q2trans1 in transverse_q:
                        for q2trans2 in transverse_q:
                            for q3plus in boson_longitudinal_q:
                                for q3trans1 in transverse_q:
                                    for q3trans2 in transverse_q:
                                        q1 = np.array(
                                            [q1plus, q1trans1, q1trans2],
                                            dtype=np.complex128,
                                        )
                                        q2 = np.array(
                                            [q2plus, q2trans1, q2trans2],
                                            dtype=np.complex128,
                                        )
                                        q3 = np.array(
                                            [q3plus, q3trans1, q3trans2],
                                            dtype=np.complex128,
                                        )
                                        q4 = -1 * (
                                            q1 + q2 + q3
                                        )  # q4 is assigned by delta

                                        if (
                                            0 < np.abs(q4[0]) <= K
                                            and np.abs(q4[1]) <= Kp
                                            and np.abs(q4[2]) <= Kp
                                        ):  # make sure k4 < K,Kp

                                            psi1 = heaviside(-q1[0]) * udag(
                                                p=-q1, m=mq, h=lambda_1
                                            ) + heaviside(
                                                q1[0] * vdag(p=q1, m=mq, h=lambda_1)
                                            )
                                            A2 = (
                                                heaviside(q2[0])
                                                * pol_vec(p=q2, pol=sigma_2)
                                                + heaviside(-q2[0])
                                                * pol_vec(p=-q2, pol=sigma_2).conj()
                                            )
                                            A3 = (
                                                heaviside(q3[0])
                                                * pol_vec(p=q3, pol=sigma_3)
                                                + heaviside(-q3[0])
                                                * pol_vec(p=-q3, pol=sigma_3).conj()
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
                                                p=q1, m=mq, h=lambda_4
                                            ) + heaviside(-q1[0]) * v(
                                                p=-q1, m=mq, h=lambda_4
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
                                            if coeff != 0:
                                                coeff *= (
                                                    g**2
                                                    * T[a - 1][c1 - 1][c4 - 1]
                                                    * T[b - 1][c1 - 1][c4 - 1]
                                                    * 1
                                                    / (
                                                        2
                                                        * np.pi
                                                        / L
                                                        * ((q3[0] + q4[0])) ** 2
                                                    )
                                                    * (1 / ((2 * L) * (Lp) ** 2)) ** 3
                                                    / (
                                                        (2 * np.pi / L) ** 4
                                                        * (
                                                            np.abs(q1[0])
                                                            * np.abs(q2[0])
                                                            * np.abs(q3[0])
                                                            * np.abs(q4[0])
                                                        )
                                                    )
                                                )

                                                particle_operator_string = (
                                                    (
                                                        heaviside(-q1[0])
                                                        * (
                                                            "b"
                                                            + str(
                                                                quark_quantum_numbers(
                                                                    k=-q1,
                                                                    K=K,
                                                                    Kp=Kp,
                                                                    c=c1,
                                                                    h=lambda_1,
                                                                    Nc=Nc,
                                                                )
                                                            )
                                                            + "^ "
                                                        )
                                                        + heaviside(q1[0])
                                                        * (
                                                            "d"
                                                            + str(
                                                                quark_quantum_numbers(
                                                                    k=q1,
                                                                    K=K,
                                                                    Kp=Kp,
                                                                    c=c1,
                                                                    h=lambda_1,
                                                                    Nc=Nc,
                                                                )
                                                            )
                                                            + " "
                                                        )
                                                    )
                                                    + (
                                                        heaviside(q2[0])
                                                        * (
                                                            "a"
                                                            + str(
                                                                gluon_quantum_numbers(
                                                                    k=q2,
                                                                    K=K,
                                                                    Kp=Kp,
                                                                    a=a,
                                                                    pol=sigma_2,
                                                                    Nc=Nc,
                                                                )
                                                            )
                                                            + " "
                                                        )
                                                        + heaviside(-q2[0])
                                                        * (
                                                            "a"
                                                            + str(
                                                                gluon_quantum_numbers(
                                                                    k=-q2,
                                                                    K=K,
                                                                    Kp=Kp,
                                                                    a=a,
                                                                    pol=sigma_2,
                                                                    Nc=Nc,
                                                                )
                                                            )
                                                            + "^ "
                                                        )
                                                    )
                                                    + (
                                                        heaviside(q3[0])
                                                        * (
                                                            "a"
                                                            + str(
                                                                gluon_quantum_numbers(
                                                                    k=q3,
                                                                    K=K,
                                                                    Kp=Kp,
                                                                    a=b,
                                                                    pol=sigma_3,
                                                                    Nc=Nc,
                                                                )
                                                            )
                                                            + " "
                                                        )
                                                        + heaviside(-q3[0])
                                                        * (
                                                            "a"
                                                            + str(
                                                                gluon_quantum_numbers(
                                                                    k=-q3,
                                                                    K=K,
                                                                    Kp=Kp,
                                                                    a=b,
                                                                    pol=sigma_3,
                                                                    Nc=Nc,
                                                                )
                                                            )
                                                            + "^ "
                                                        )
                                                    )
                                                    + (
                                                        heaviside(q4[0])
                                                        * (
                                                            "b"
                                                            + str(
                                                                quark_quantum_numbers(
                                                                    k=q4,
                                                                    K=K,
                                                                    Kp=Kp,
                                                                    c=c4,
                                                                    h=lambda_4,
                                                                    Nc=Nc,
                                                                )
                                                            )
                                                            + "^ "
                                                        )
                                                        + heaviside(-q4[0])
                                                        * (
                                                            "d"
                                                            + str(
                                                                quark_quantum_numbers(
                                                                    k=-q4,
                                                                    K=K,
                                                                    Kp=Kp,
                                                                    c=c4,
                                                                    h=lambda_4,
                                                                    Nc=Nc,
                                                                )
                                                            )
                                                            + " "
                                                        )
                                                    )
                                                )
                                                fixed_qns_dict[
                                                    particle_operator_string
                                                ] = coeff

    return fixed_qns_dict


@nb.njit(
    ## output types
    nb.types.ListType(nb.types.DictType(nb.types.unicode_type, nb.types.complex128))
    ##input types
    (nb.float64, nb.float64, nb.float64, nb.float64, nb.int64),
    ## other options
    fastmath=True,
)
def fermion_exch_inst_term(g: float, mq: float, K: float, Kp: float, Nc: int = 3):
    gluon_polarizations = [1, -1]
    gluon_colors = np.arange(1, Nc**2 - 1 + 1, 1)
    quark_helicities = [1, -1]
    quark_colors = np.arange(1, Nc + 1, 1)

    fermion_exch_inst_term_list = nb.typed.List()

    for a in gluon_colors:
        for b in gluon_colors:
            for c1 in quark_colors:
                for c4 in quark_colors:
                    for lambda_1 in quark_helicities:
                        for lambda_4 in quark_helicities:
                            for sigma_2 in gluon_polarizations:
                                for sigma_3 in gluon_polarizations:

                                    if (
                                        T[a - 1][c1 - 1][c4 - 1]
                                        * T[b - 1][c1 - 1][c4 - 1]
                                        != 0
                                    ):
                                        fermion_exch_inst_term_list.append(
                                            fixed_fermion_exch_inst_term(
                                                lambda_1=lambda_1,
                                                sigma_2=sigma_2,
                                                sigma_3=sigma_3,
                                                lambda_4=lambda_4,
                                                a=a,
                                                c1=c1,
                                                c4=c4,
                                                b=b,
                                                g=g,
                                                mq=mq,
                                                K=K,
                                                Kp=Kp,
                                                Nc=Nc,
                                            )
                                        )

    return fermion_exch_inst_term_list

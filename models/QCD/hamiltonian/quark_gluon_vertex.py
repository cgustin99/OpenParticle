import numpy as np
from openparticle.full_dlcq import *
import openparticle as op
from gell_mann import T
from itertools import product
from collections import defaultdict


def heaviside(x: float, y: float = 0):
    return (x > y).astype(int)


def fixed_quark_gluon_vertex_term(
    lambda_1: int,
    lambda_2: int,
    sigma: int,
    c1: int,
    c2: int,
    cg: int,
    mq: float,
    g: float,
    K: int,
    Kp: int,
    Nc: int = 3,
):

    fixed_qns_dict = {}

    fermion_lim = K - ((K - 0.5) % 1)
    boson_lim = int(K)
    L = 2 * np.pi * K
    Lp = 2 * np.pi * Kp

    fermion_longitudinal_q = np.arange(-fermion_lim, fermion_lim + 1, 1)
    boson_longitudinal_q = np.array(
        [i for i in np.arange(-boson_lim, boson_lim + 1, 1) if i != 0]
    )
    transverse_q = np.array([i for i in range(-Kp, Kp + 1)])

    q1_range = product(fermion_longitudinal_q, transverse_q, transverse_q)
    q3_range = product(boson_longitudinal_q, transverse_q, transverse_q)

    for i in product(q1_range, q3_range):
        q1 = np.array(i[0])
        q3 = np.array(i[1])
        q2 = -1 * (q3 + q1)  # q2 is assigned by delta

        if (
            np.abs(q2[0]) <= K and np.abs(q2[1]) <= Kp and np.abs(q2[2]) <= Kp
        ):  # make sure k2 < K,Kp

            e = pol_vec(p=q3, pol=sigma)
            estar = pol_vec(p=-q3, pol=sigma).conj()

            e_slash = 0.5 * gamma_plus.dot(e[1]) - gamma1.dot(e[2]) - gamma2.dot(e[3])
            estar_slash = (
                0.5 * gamma_plus.dot(estar[1])
                - gamma1.dot(estar[2])
                - gamma2.dot(estar[3])
            )

            A = heaviside(-q1[0]) * udag(p=-q1, m=mq, h=lambda_1) + heaviside(
                q1[0]
            ) * vdag(p=q1, m=mq, h=lambda_1)
            B = heaviside(q3[0]) * e_slash + heaviside(-q3[0]) * estar_slash
            C = heaviside(q2[0]) * u(p=q2, m=mq, h=lambda_2) + heaviside(-q2[0]) * v(
                p=-q2, m=mq, h=lambda_2
            )

            coeff = (
                g
                * T[cg - 1][c1 - 1][c2 - 1]
                * (1 / ((2 * L) * (Lp) ** 2)) ** 3
                * (A.dot(gamma0.dot(B.dot(C))))[0][0]
            )
            if coeff != 0:
                particle_operator_string = ""
                string_1 = (
                    heaviside(-q1[0])
                    * heaviside(q2[0])
                    * heaviside(q3[0])
                    * (
                        "b"
                        + str(
                            quark_quantum_numbers(
                                k=-q1[0],
                                kp=-q1[1:3],
                                K=K,
                                Kp=Kp,
                                c=c1,
                                h=lambda_1,
                                Nc=Nc,
                            )
                        )
                        + "^ "
                        + "b"
                        + str(
                            quark_quantum_numbers(
                                k=q2[0], kp=q2[1:3], K=K, Kp=Kp, c=c2, h=lambda_2, Nc=Nc
                            )
                        )
                        + " "
                        + "a"
                        + str(
                            gluon_quantum_numbers(
                                k=q3[0], kp=q3[1:3], K=K, Kp=Kp, a=cg, pol=sigma, Nc=Nc
                            )
                        )
                    )
                )
                string_2 = (
                    heaviside(-q1[0])
                    * heaviside(-q2[0])
                    * heaviside(q3[0])
                    * (
                        "b"
                        + str(
                            quark_quantum_numbers(
                                k=-q1[0],
                                kp=-q1[1:3],
                                K=K,
                                Kp=Kp,
                                c=c1,
                                h=lambda_1,
                                Nc=Nc,
                            )
                        )
                        + "^ "
                        + "d"
                        + str(
                            quark_quantum_numbers(
                                k=-q2[0],
                                kp=-q2[1:3],
                                K=K,
                                Kp=Kp,
                                c=c2,
                                h=lambda_2,
                                Nc=Nc,
                            )
                        )
                        + "^ "
                        + "a"
                        + str(
                            gluon_quantum_numbers(
                                k=q3[0], kp=q3[1:3], K=K, Kp=Kp, a=cg, pol=sigma, Nc=Nc
                            )
                        )
                    )
                )

                string_3 = (
                    heaviside(-q1[0])
                    * heaviside(q2[0])
                    * heaviside(-q3[0])
                    * (
                        "b"
                        + str(
                            quark_quantum_numbers(
                                k=-q1[0],
                                kp=-q1[1:3],
                                K=K,
                                Kp=Kp,
                                c=c1,
                                h=lambda_1,
                                Nc=Nc,
                            )
                        )
                        + "^ "
                        + "b"
                        + str(
                            quark_quantum_numbers(
                                k=q2[0], kp=q2[1:3], K=K, Kp=Kp, c=c2, h=lambda_2, Nc=Nc
                            )
                        )
                        + " "
                        + "a"
                        + str(
                            gluon_quantum_numbers(
                                k=-q3[0],
                                kp=-q3[1:3],
                                K=K,
                                Kp=Kp,
                                a=cg,
                                pol=sigma,
                                Nc=Nc,
                            )
                        )
                        + "^"
                    )
                )

                string_4 = (
                    heaviside(q1[0])
                    * heaviside(-q2[0])
                    * heaviside(q3[0])
                    * (
                        "d"
                        + str(
                            quark_quantum_numbers(
                                k=q1[0], kp=q1[1:3], K=K, Kp=Kp, c=c1, h=lambda_1, Nc=Nc
                            )
                        )
                        + " "
                        + "d"
                        + str(
                            quark_quantum_numbers(
                                k=-q2[0],
                                kp=-q2[1:3],
                                K=K,
                                Kp=Kp,
                                c=c2,
                                h=lambda_2,
                                Nc=Nc,
                            )
                        )
                        + "^ "
                        + "a"
                        + str(
                            gluon_quantum_numbers(
                                k=q3[0], kp=q3[1:3], K=K, Kp=Kp, a=cg, pol=sigma, Nc=Nc
                            )
                        )
                    )
                )

                string_5 = (
                    heaviside(q1[0])
                    * heaviside(-q2[0])
                    * heaviside(-q3[0])
                    * (
                        "d"
                        + str(
                            quark_quantum_numbers(
                                k=q1[0], kp=q1[1:3], K=K, Kp=Kp, c=c1, h=lambda_1, Nc=Nc
                            )
                        )
                        + " "
                        + "d"
                        + str(
                            quark_quantum_numbers(
                                k=-q2[0],
                                kp=-q2[1:3],
                                K=K,
                                Kp=Kp,
                                c=c2,
                                h=lambda_2,
                                Nc=Nc,
                            )
                        )
                        + "^ "
                        + "a"
                        + str(
                            gluon_quantum_numbers(
                                k=-q3[0],
                                kp=-q3[1:3],
                                K=K,
                                Kp=Kp,
                                a=cg,
                                pol=sigma,
                                Nc=Nc,
                            )
                        )
                    )
                )

                string_6 = (
                    heaviside(q1[0])
                    * heaviside(q2[0])
                    * heaviside(-q3[0])
                    * (
                        "d"
                        + str(
                            quark_quantum_numbers(
                                k=q1[0], kp=q1[1:3], K=K, Kp=Kp, c=c1, h=lambda_1, Nc=Nc
                            )
                        )
                        + " "
                        + "b"
                        + str(
                            quark_quantum_numbers(
                                k=q2[0], kp=q2[1:3], K=K, Kp=Kp, c=c2, h=lambda_2, Nc=Nc
                            )
                        )
                        + " "
                        + "a"
                        + str(
                            gluon_quantum_numbers(
                                k=-q3[0],
                                kp=-q3[1:3],
                                K=K,
                                Kp=Kp,
                                a=cg,
                                pol=sigma,
                                Nc=Nc,
                            )
                        )
                        + "^"
                    )
                )

                particle_operator_string = (
                    string_1 + string_2 + string_3 + string_4 + string_5 + string_6
                )

                fixed_qns_dict[particle_operator_string] = coeff
    return fixed_qns_dict


def quark_gluon_vertex_term(g: float, mq: float, K: float, Kp: float, Nc: int = 3):
    quark1_helicities = [1, -1]
    quark2_helicities = [1, -1]
    gluon_polarizations = [1, -1]
    quark1_colors = np.arange(1, Nc + 1, 1)
    quark2_colors = np.arange(1, Nc + 1, 1)
    gluon_colors = np.arange(1, Nc**2 - 1 + 1, 1)

    quark_gluon_vertex_term_list = []

    for qns in product(
        quark1_helicities,
        quark2_helicities,
        gluon_polarizations,
        quark1_colors,
        quark2_colors,
        gluon_colors,
    ):
        quark_gluon_vertex_term_list.append(
            fixed_quark_gluon_vertex_term(
                lambda_1=qns[0],
                lambda_2=qns[1],
                sigma=qns[2],
                c1=qns[3],
                c2=qns[4],
                cg=qns[5],
                mq=mq,
                g=g,
                K=K,
                Kp=Kp,
                Nc=Nc,
            )
        )

    quark_gluon_vertex_term_dictionary = defaultdict(int)

    for d in quark_gluon_vertex_term_list:
        for key, value in d.items():
            quark_gluon_vertex_term_dictionary[key] += value

    quark_gluon_vertex_term_dictionary = dict(quark_gluon_vertex_term_dictionary)
    return quark_gluon_vertex_term_dictionary


print(quark_gluon_vertex_term(g=1, mq=1, K=2, Kp=1))

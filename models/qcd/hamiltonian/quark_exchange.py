import numpy as np
from openparticle.full_dlcq import *
import openparticle as op
from gell_mann import T, f
from itertools import product
from collections import defaultdict


def heaviside(x: float, y: float = 0):
    return (x > y).astype(int)


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
    K: int,
    Kp: int,
    Nc: int = 3,
):

    fixed_qns_dict = {}

    boson_lim = int(K)
    fermion_lim = K - ((K - 0.5) % 1)

    L = 2 * np.pi * K
    Lp = 2 * np.pi * Kp

    boson_longitudinal_q = np.array(
        [i for i in np.arange(-boson_lim, boson_lim + 1, 1) if i != 0]
    )
    fermion_longitudinal_q = np.array(
        [i for i in np.arange(-fermion_lim, fermion_lim + 1, 1)]
    )
    transverse_q = np.array([i for i in range(-Kp, Kp + 1)])

    q1_range = product(fermion_longitudinal_q, transverse_q, transverse_q)
    q2_range = product(boson_longitudinal_q, transverse_q, transverse_q)
    q3_range = product(boson_longitudinal_q, transverse_q, transverse_q)

    for i in product(q1_range, q2_range, q3_range):
        q1 = np.array(i[0])
        q2 = np.array(i[1])
        q3 = np.array(i[2])
        q4 = -1 * (q1 + q2 + q3)  # q4 is assigned by delta

        if 0 < np.abs(q4[0]) <= K and np.abs(q4[1]) <= Kp and np.abs(q4[2]) <= Kp:
            psi1 = heaviside(-q1[0]) * udag(p=-q1, m=mq, h=lambda_1) + heaviside(
                q1[0] * vdag(p=q1, m=mq, h=lambda_1)
            )
            A2 = (
                heaviside(q2[0]) * pol_vec(p=q2, pol=sigma_2)
                + heaviside(-q2[0]) * pol_vec(p=-q2, pol=sigma_2).conj()
            )
            A3 = (
                heaviside(q3[0]) * pol_vec(p=q3, pol=sigma_3)
                + heaviside(-q3[0]) * pol_vec(p=-q3, pol=sigma_3).conj()
            )
            A2slash = (
                0.5 * gamma_plus.dot(A2[1]) - gamma1.dot(A2[2]) - gamma2.dot(A2[3])
            )
            A3slash = (
                0.5 * gamma_plus.dot(A3[1]) - gamma1.dot(A3[2]) - gamma2.dot(A3[3])
            )
            psi4 = heaviside(q4[0]) * u(p=q1, m=mq, h=lambda_4) + heaviside(-q1[0]) * v(
                p=-q1, m=mq, h=lambda_4
            )

            coeff = psi1.dot(
                gamma0.dot(A2slash.dot(gamma_plus.dot(A3slash.dot(psi4))))
            )[0][0]
            if coeff != 0:
                coeff *= (
                    g**2
                    * T[a - 1][c1 - 1][c4 - 1]
                    * T[b - 1][c1 - 1][c4 - 1]
                    * 1
                    / (2 * np.pi / L * ((q3[0] + q4[0])) ** 2)
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
                                    k=-q1, K=K, Kp=Kp, c=c1, h=lambda_1
                                )
                            )
                            + "^ "
                        )
                        + heaviside(q1[0])
                        * (
                            "d"
                            + str(
                                quark_quantum_numbers(
                                    k=q1, K=K, Kp=Kp, c=c1, h=lambda_1
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
                                    k=q2, K=K, Kp=Kp, a=a, pol=sigma_2
                                )
                            )
                            + " "
                        )
                        + heaviside(-q2[0])
                        * (
                            "a"
                            + str(
                                gluon_quantum_numbers(
                                    k=-q2, K=K, Kp=Kp, a=a, pol=sigma_2
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
                                    k=q3, K=K, Kp=Kp, a=b, pol=sigma_3
                                )
                            )
                            + " "
                        )
                        + heaviside(-q3[0])
                        * (
                            "a"
                            + str(
                                gluon_quantum_numbers(
                                    k=-q3, K=K, Kp=Kp, a=b, pol=sigma_3
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
                                    k=q4, K=K, Kp=Kp, c=c4, h=lambda_4
                                )
                            )
                            + "^ "
                        )
                        + heaviside(-q4[0])
                        * (
                            "d"
                            + str(
                                quark_quantum_numbers(
                                    k=-q4, K=K, Kp=Kp, c=c4, h=lambda_4
                                )
                            )
                            + " "
                        )
                    )
                )
                fixed_qns_dict[particle_operator_string] = coeff

    return fixed_qns_dict


def fermion_exch_inst_term(g: float, mq: float, K: int, Kp: int, Nc: int = 3):
    gluon_polarizations = [1, -1]
    gluon_colors = np.arange(1, Nc**2 - 1 + 1, 1)
    quark_helicities = [1, -1]
    quark_colors = np.arange(1, Nc + 1, 1)

    fermion_exch_inst_term_list = []

    for qns in product(
        quark_helicities,
        quark_helicities,
        gluon_polarizations,
        gluon_polarizations,
        quark_colors,
        quark_colors,
        gluon_colors,
        gluon_colors,
    ):
        c1, c4, a, b = qns[4], qns[5], qns[6], qns[7]

        if T[a - 1][c1 - 1][c4 - 1] * T[b - 1][c1 - 1][c4 - 1] != 0:
            fermion_exch_inst_term_list.append(
                fixed_fermion_exch_inst_term(
                    lambda_1=qns[0],
                    sigma_2=qns[1],
                    sigma_3=qns[2],
                    lambda_4=qns[3],
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

    fermion_exch_inst_term_dictionary = defaultdict(int)

    for d in fermion_exch_inst_term_list:
        for key, value in d.items():
            fermion_exch_inst_term_dictionary[key] += value

    fermion_exch_inst_term_dictionary = dict(fermion_exch_inst_term_dictionary)
    return fermion_exch_inst_term_dictionary

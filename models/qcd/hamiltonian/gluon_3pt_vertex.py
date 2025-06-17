import numpy as np
from openparticle.full_dlcq import *
import openparticle as op
from gell_mann import T, f
from itertools import product
from collections import defaultdict


def heaviside(x: float, y: float = 0):
    return (x > y).astype(int)


def fixed_gluon_3pt_vertex_term(
    sigma_1: int,
    sigma_2: int,
    sigma_3: int,
    a: int,
    b: int,
    c: int,
    g: float,
    K: int,
    Kp: int,
    Nc: int = 3,
):

    fixed_qns_dict = {}

    boson_lim = int(K)
    L = 2 * np.pi * K
    Lp = 2 * np.pi * Kp

    boson_longitudinal_q = np.array(
        [i for i in np.arange(-boson_lim, boson_lim + 1, 1) if i != 0]
    )
    transverse_q = np.array([i for i in range(-Kp, Kp + 1)])

    q1_range = product(boson_longitudinal_q, transverse_q, transverse_q)
    q2_range = product(boson_longitudinal_q, transverse_q, transverse_q)

    for i in product(q1_range, q2_range):
        q1 = np.array(i[0])
        q2 = np.array(i[1])
        q3 = -1 * (q1 + q2)  # q3 is assigned by delta

        coeff = 0

        if (
            0 < np.abs(q3[0]) <= K and np.abs(q3[1]) <= Kp and np.abs(q3[2]) <= Kp
        ):  # make sure k3 < K,Kp

            Aone = (
                heaviside(q1[0]) * pol_vec(p=q1, pol=sigma_1)
                + heaviside(-q1[0]) * pol_vec(p=q1, pol=sigma_1).conj()
            )

            Atwo = (
                heaviside(q2[0]) * pol_vec(p=q2, pol=sigma_2)
                + heaviside(-q2[0]) * pol_vec(p=q2, pol=sigma_2).conj()
            )

            Athree = (
                heaviside(q3[0]) * pol_vec(p=q3, pol=sigma_3)
                + heaviside(-q3[0]) * pol_vec(p=q3, pol=sigma_3).conj()
            )

            # coeff += q2^+ (A1A1 + A2A2)A- - q2^1(A1A1 + A2A2)A1 - q2^2(A1A1 + A2A2)A2
            coeff = (
                q2[0] * (Aone[2] * Atwo[2] + Aone[3] * Atwo[3]) * Athree[1]
                - q2[1] * (Aone[2] * Atwo[2] + Aone[3] * Atwo[3]) * Athree[2]
                - q2[2] * (Aone[2] * Atwo[2] + Aone[3] * Atwo[3]) * Athree[3]
            )

            coeff *= (
                -1j
                * g
                * f(a, b, c)
                * (1 / ((2 * L) * (Lp) ** 2)) ** 3
                / (2 * np.pi / L * (np.abs(q1[0]) * np.abs(q2[0]) * np.abs(q3[0])))
            )

            if coeff != 0:
                particle_operator_string = ""
                string_1 = (
                    heaviside(q1[0])
                    * heaviside(q2[0])
                    * heaviside(-q3[0])
                    * (
                        "a"
                        + str(
                            gluon_quantum_numbers(
                                k=q1,
                                K=K,
                                Kp=Kp,
                                a=b,
                                pol=sigma_1,
                                Nc=Nc,
                            )
                        )
                        + " "
                        + "a"
                        + str(
                            gluon_quantum_numbers(
                                k=q2,
                                K=K,
                                Kp=Kp,
                                a=c,
                                pol=sigma_2,
                                Nc=Nc,
                            )
                        )
                        + " "
                        + "a"
                        + str(
                            gluon_quantum_numbers(
                                k=-q3,
                                K=K,
                                Kp=Kp,
                                a=a,
                                pol=sigma_3,
                                Nc=Nc,
                            )
                        )
                        + "^"
                    )
                )

                string_2 = (
                    heaviside(q1[0])
                    * heaviside(-q2[0])
                    * heaviside(q3[0])
                    * (
                        "a"
                        + str(
                            gluon_quantum_numbers(
                                k=q1,
                                K=K,
                                Kp=Kp,
                                a=b,
                                pol=sigma_1,
                                Nc=Nc,
                            )
                        )
                        + " "
                        + "a"
                        + str(
                            gluon_quantum_numbers(
                                k=-q2,
                                K=K,
                                Kp=Kp,
                                a=c,
                                pol=sigma_2,
                                Nc=Nc,
                            )
                        )
                        + "^ "
                        + "a"
                        + str(
                            gluon_quantum_numbers(
                                k=q3,
                                K=K,
                                Kp=Kp,
                                a=a,
                                pol=sigma_3,
                                Nc=Nc,
                            )
                        )
                    )
                )

                string_3 = (
                    heaviside(q1[0])
                    * heaviside(-q2[0])
                    * heaviside(-q3[0])
                    * (
                        "a"
                        + str(
                            gluon_quantum_numbers(
                                k=q1,
                                K=K,
                                Kp=Kp,
                                a=b,
                                pol=sigma_1,
                                Nc=Nc,
                            )
                        )
                        + " "
                        + "a"
                        + str(
                            gluon_quantum_numbers(
                                k=-q2,
                                K=K,
                                Kp=Kp,
                                a=c,
                                pol=sigma_2,
                                Nc=Nc,
                            )
                        )
                        + "^ "
                        + "a"
                        + str(
                            gluon_quantum_numbers(
                                k=-q3,
                                K=K,
                                Kp=Kp,
                                a=a,
                                pol=sigma_3,
                                Nc=Nc,
                            )
                        )
                        + "^"
                    )
                )

                string_4 = (
                    heaviside(-q1[0])
                    * heaviside(q2[0])
                    * heaviside(q3[0])
                    * (
                        "a"
                        + str(
                            gluon_quantum_numbers(
                                k=-q1,
                                K=K,
                                Kp=Kp,
                                a=b,
                                pol=sigma_1,
                                Nc=Nc,
                            )
                        )
                        + "^ "
                        + "a"
                        + str(
                            gluon_quantum_numbers(
                                k=q2,
                                K=K,
                                Kp=Kp,
                                a=c,
                                pol=sigma_2,
                                Nc=Nc,
                            )
                        )
                        + " "
                        + "a"
                        + str(
                            gluon_quantum_numbers(
                                k=q3,
                                K=K,
                                Kp=Kp,
                                a=a,
                                pol=sigma_3,
                                Nc=Nc,
                            )
                        )
                    )
                )

                string_5 = (
                    heaviside(-q1[0])
                    * heaviside(-q2[0])
                    * heaviside(q3[0])
                    * (
                        "a"
                        + str(
                            gluon_quantum_numbers(
                                k=-q1,
                                K=K,
                                Kp=Kp,
                                a=b,
                                pol=sigma_1,
                                Nc=Nc,
                            )
                        )
                        + "^ "
                        + "a"
                        + str(
                            gluon_quantum_numbers(
                                k=-q2,
                                K=K,
                                Kp=Kp,
                                a=c,
                                pol=sigma_2,
                                Nc=Nc,
                            )
                        )
                        + "^ "
                        + "a"
                        + str(
                            gluon_quantum_numbers(
                                k=q3,
                                K=K,
                                Kp=Kp,
                                a=a,
                                pol=sigma_3,
                                Nc=Nc,
                            )
                        )
                    )
                )

                string_6 = (
                    heaviside(-q1[0])
                    * heaviside(q2[0])
                    * heaviside(-q3[0])
                    * (
                        "a"
                        + str(
                            gluon_quantum_numbers(
                                k=-q1,
                                K=K,
                                Kp=Kp,
                                a=b,
                                pol=sigma_1,
                                Nc=Nc,
                            )
                        )
                        + "^ "
                        + "a"
                        + str(
                            gluon_quantum_numbers(
                                k=q2,
                                K=K,
                                Kp=Kp,
                                a=c,
                                pol=sigma_2,
                                Nc=Nc,
                            )
                        )
                        + " "
                        + "a"
                        + str(
                            gluon_quantum_numbers(
                                k=-q3,
                                K=K,
                                Kp=Kp,
                                a=a,
                                pol=sigma_3,
                                Nc=Nc,
                            )
                        )
                        + "^ "
                    )
                )

                particle_operator_string = (
                    string_1 + string_2 + string_3 + string_4 + string_5 + string_6
                )

                fixed_qns_dict[particle_operator_string] = coeff
    return fixed_qns_dict


def gluon_3pt_vertex_term(g: float, K: float, Kp: float, Nc: int = 3):
    gluon_polarizations = [1, -1]
    gluon1_colors = np.arange(1, Nc**2 - 1 + 1, 1)
    gluon2_colors = np.arange(1, Nc**2 - 1 + 1, 1)
    gluon3_colors = np.arange(1, Nc**2 - 1 + 1, 1)

    gluon_3pt_vertex_term_list = []

    for qns in product(
        gluon_polarizations,
        gluon_polarizations,
        gluon_polarizations,
        gluon1_colors,
        gluon2_colors,
        gluon3_colors,
    ):
        fabc = f(qns[3], qns[4], qns[5])
        if fabc != 0:
            gluon_3pt_vertex_term_list.append(
                fixed_gluon_3pt_vertex_term(
                    sigma_1=qns[0],
                    sigma_2=qns[1],
                    sigma_3=qns[2],
                    a=qns[3],
                    b=qns[4],
                    c=qns[5],
                    g=g,
                    K=K,
                    Kp=Kp,
                    Nc=Nc,
                )
            )

    gluon_3pt_vertex_term_dictionary = defaultdict(int)

    for d in gluon_3pt_vertex_term_list:
        for key, value in d.items():
            gluon_3pt_vertex_term_dictionary[key] += value

    gluon_3pt_vertex_term_dictionary = dict(gluon_3pt_vertex_term_dictionary)
    return gluon_3pt_vertex_term_dictionary


print(gluon_3pt_vertex_term(g=1, K=2, Kp=1))

import openparticle as op
from openparticle.full_dlcq import *
from gell_mann import T
import numpy as np
from itertools import product


def heaviside(x: float, y: float = 0):
    return (x > y).astype(int)


def fixed_free_quark_hamiltonian(
    color: int, helicity: int, K: int, Kp: int, mq: float, Nc: int = 3
):
    fixed_qns_dict = {}
    fermion_lim = K - ((K - 0.5) % 1)

    L = 2 * np.pi * K
    Lp = 2 * np.pi * Kp

    fermion_longitudinal_q = np.array(
        [i for i in np.arange(-fermion_lim, fermion_lim + 1, 1)]
    )
    transverse_q = np.array([i for i in range(-Kp, Kp + 1)])

    qrange = product(fermion_longitudinal_q, transverse_q, transverse_q)

    for i in qrange:
        q = np.array(i)
        psi_dag = heaviside(q[0]) * udag(p=q, m=mq, h=helicity) + heaviside(
            -q[0]
        ) * vdag(p=-q, m=mq, h=helicity)
        psi = heaviside(q[0]) * u(p=q, m=mq, h=helicity) + heaviside(-q[0]) * v(
            p=-q, m=mq, h=helicity
        )
        coeff = psi_dag.dot(gamma0.dot(gamma_plus.dot(psi)))[0][0]
        coeff *= (
            0.5 * (1 / ((2 * L) * (Lp) ** 2)) * ((q[1:3].dot(q[1:3]) + mq**2) / q[0])
        )
        particle_operator_string = (
            heaviside(q[0])
            * (
                "b"
                + str(
                    quark_quantum_numbers(k=q, K=K, Kp=Kp, c=color, h=helicity, Nc=Nc)
                )
                + "^ "
            )
            + heaviside(-q[0])
            * (
                "d"
                + str(
                    quark_quantum_numbers(k=-q, K=K, Kp=Kp, c=color, h=helicity, Nc=Nc)
                )
                + " "
            )
        ) + (
            heaviside(q[0])
            * (
                "b"
                + str(
                    quark_quantum_numbers(k=q, K=K, Kp=Kp, c=color, h=helicity, Nc=Nc)
                )
                + " "
            )
            + heaviside(-q[0])
            * (
                "d"
                + str(
                    quark_quantum_numbers(k=-q, K=K, Kp=Kp, c=color, h=helicity, Nc=Nc)
                )
                + "^ "
            )
        )
        fixed_qns_dict[particle_operator_string] = coeff

    return fixed_qns_dict


def free_quark_hamiltonian(K, Kp, mq, Nc=3):
    quark_helicities = [1, -1]
    quark_colors = np.arange(1, Nc + 1, 1)

    free_quark_term_list = []
    for qns in product(quark_helicities, quark_colors):
        free_quark_term_list.append(
            fixed_free_quark_hamiltonian(
                color=qns[1], helicity=qns[0], K=K, Kp=Kp, mq=mq, Nc=Nc
            )
        )
    return free_quark_term_list


def fixed_free_gluon_hamiltonian(
    color: int, polarization: int, K: int, Kp: int, mg: float, Nc: int = 3
):
    fixed_qns_dict = {}
    boson_lim = int(K)
    L = 2 * np.pi * K
    Lp = 2 * np.pi * Kp

    boson_longitudinal_q = np.array(
        [i for i in np.arange(-boson_lim, boson_lim + 1, 1) if i != 0]
    )
    transverse_q = np.array([i for i in range(-Kp, Kp + 1)])

    qrange = product(boson_longitudinal_q, transverse_q, transverse_q)

    for i in qrange:
        q = np.array(i)

        Aplus = (
            heaviside(q[0]) * pol_vec(p=q, pol=polarization)
            + heaviside(-q[0]) * pol_vec(p=-q, pol=polarization).conj()
        )
        Aminus = (
            heaviside(-q[0]) * pol_vec(p=-q, pol=polarization)
            + heaviside(q[0]) * pol_vec(p=q, pol=polarization).conj()
        )
        coeff = -1 * Aminus[2] * Aplus[2] - 1 * Aminus[3] * Aplus[3]
        print(coeff)
        coeff *= 0.5 * (1 / ((2 * L) * (Lp) ** 2)) * ((q[1:3].dot(q[1:3]) + mg**2))
        particle_operator_string = (
            heaviside(-q[0])
            * (
                "a"
                + str(
                    gluon_quantum_numbers(
                        k=-q, K=K, Kp=Kp, a=color, pol=polarization, Nc=Nc
                    )
                )
                + " "
            )
            + heaviside(q[0])
            * (
                "a"
                + str(
                    gluon_quantum_numbers(
                        k=q, K=K, Kp=Kp, a=color, pol=polarization, Nc=Nc
                    )
                )
                + "^ "
            )
        ) + (
            heaviside(q[0])
            * (
                "a"
                + str(
                    gluon_quantum_numbers(
                        k=q, K=K, Kp=Kp, a=color, pol=polarization, Nc=Nc
                    )
                )
                + ""
            )
            + heaviside(-q[0])
            * (
                "a"
                + str(
                    gluon_quantum_numbers(
                        k=-q, K=K, Kp=Kp, a=color, pol=polarization, Nc=Nc
                    )
                )
                + "^"
            )
        )
        fixed_qns_dict[particle_operator_string] = coeff

    return fixed_qns_dict


def free_gluon_hamiltonian(K, Kp, mg, Nc=3):
    gluon_helicities = [1, -1]
    gluon_colors = np.arange(1, Nc**2 - 1 + 1, 1)

    free_gluon_term_list = []
    for qns in product(gluon_helicities, gluon_colors):
        free_gluon_term_list.append(
            fixed_free_gluon_hamiltonian(
                color=qns[1], polarization=qns[0], K=K, Kp=Kp, mg=mg, Nc=Nc
            )
        )
    return free_gluon_term_list

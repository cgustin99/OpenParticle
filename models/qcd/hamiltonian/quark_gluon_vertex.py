import openparticle as op
from openparticle.full_dlcq import *
from models.qcd.hamiltonian.gell_mann import T
import numpy as np
import numba as nb


@nb.njit
def heaviside(x: complex, y: complex = 0) -> int:
    return 1 if x.real > y.real else 0


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
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.int64,
    ),
    ## other options
    fastmath=True,
)
def fixed_quark_gluon_vertex_term(
    lambda_1: int,
    lambda_2: int,
    sigma: int,
    c1: int,
    c2: int,
    cg: int,
    mq: float,
    g: float,
    K: float,
    Kp: float,
    Nc: int = 3,
):

    fixed_qns_dict = nb.typed.Dict.empty(
        key_type=nb.types.unicode_type, value_type=nb.types.complex128
    )

    fermion_lim = K - ((K - 0.5) % 1)
    boson_lim = int(K)
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
                for q3plus in boson_longitudinal_q:
                    for q3trans1 in transverse_q:
                        for q3trans2 in transverse_q:
                            q1 = np.array(
                                [q1plus, q1trans1, q1trans2], dtype=np.complex128
                            )
                            q3 = np.array(
                                [q3plus, q3trans1, q3trans2], dtype=np.complex128
                            )
                            q2 = -1 * (q3 + q1)  # q2 is assigned by delta

                            if (
                                np.abs(q2[0]) <= K
                                and np.abs(q2[1]) <= Kp
                                and np.abs(q2[2]) <= Kp
                            ):  # make sure k2 < K,Kp

                                e = pol_vec(p=q3, pol=sigma)
                                estar = pol_vec(p=-q3, pol=sigma).conj()

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

                                A = heaviside(-q1[0]) * udag(
                                    p=-q1, m=mq, h=lambda_1
                                ) + heaviside(q1[0]) * vdag(p=q1, m=mq, h=lambda_1)
                                B = (
                                    heaviside(q3[0]) * e_slash
                                    + heaviside(-q3[0]) * estar_slash
                                )
                                C = heaviside(q2[0]) * u(
                                    p=q2, m=mq, h=lambda_2
                                ) + heaviside(-q2[0]) * v(p=-q2, m=mq, h=lambda_2)

                                coeff = (
                                    g
                                    * T[cg - 1][c1 - 1][c2 - 1]
                                    * (1 / ((2 * L) * (Lp) ** 2)) ** 3
                                    * (A.dot(gamma0.dot(B.dot(C))))
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
                                                    k=-q1,
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
                                                    k=q2,
                                                    K=K,
                                                    Kp=Kp,
                                                    c=c2,
                                                    h=lambda_2,
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
                                                    a=cg,
                                                    pol=sigma,
                                                    Nc=Nc,
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
                                                    k=-q1,
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
                                                    k=-q2,
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
                                                    k=q3,
                                                    K=K,
                                                    Kp=Kp,
                                                    a=cg,
                                                    pol=sigma,
                                                    Nc=Nc,
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
                                                    k=-q1,
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
                                                    k=q2,
                                                    K=K,
                                                    Kp=Kp,
                                                    c=c2,
                                                    h=lambda_2,
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
                                                    k=q1,
                                                    K=K,
                                                    Kp=Kp,
                                                    c=c1,
                                                    h=lambda_1,
                                                    Nc=Nc,
                                                )
                                            )
                                            + " "
                                            + "d"
                                            + str(
                                                quark_quantum_numbers(
                                                    k=-q2,
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
                                                    k=q3,
                                                    K=K,
                                                    Kp=Kp,
                                                    a=cg,
                                                    pol=sigma,
                                                    Nc=Nc,
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
                                                    k=q1,
                                                    K=K,
                                                    Kp=Kp,
                                                    c=c1,
                                                    h=lambda_1,
                                                    Nc=Nc,
                                                )
                                            )
                                            + " "
                                            + "d"
                                            + str(
                                                quark_quantum_numbers(
                                                    k=-q2,
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
                                                    k=-q3,
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
                                                    k=q1,
                                                    K=K,
                                                    Kp=Kp,
                                                    c=c1,
                                                    h=lambda_1,
                                                    Nc=Nc,
                                                )
                                            )
                                            + " "
                                            + "b"
                                            + str(
                                                quark_quantum_numbers(
                                                    k=q2,
                                                    K=K,
                                                    Kp=Kp,
                                                    c=c2,
                                                    h=lambda_2,
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
                                                    a=cg,
                                                    pol=sigma,
                                                    Nc=Nc,
                                                )
                                            )
                                            + "^"
                                        )
                                    )

                                    particle_operator_string = (
                                        string_1
                                        + string_2
                                        + string_3
                                        + string_4
                                        + string_5
                                        + string_6
                                    )

                                    fixed_qns_dict[particle_operator_string] = coeff

    return fixed_qns_dict


@nb.njit(
    ## output types
    nb.types.ListType(nb.types.DictType(nb.types.unicode_type, nb.types.complex128))
    ##input types
    (nb.float64, nb.float64, nb.float64, nb.float64, nb.int64),
    ## other options
    fastmath=True,
)
def quark_gluon_vertex_term(g: float, mq: float, K: float, Kp: float, Nc: int = 3):
    quark_helicities = [1, -1]
    gluon_polarizations = [1, -1]
    quark_colors = np.arange(1, Nc + 1, 1)
    gluon_colors = np.arange(1, Nc**2 - 1 + 1, 1)

    quark_gluon_vertex_term_list = nb.typed.List()

    for lambda_1 in quark_helicities:
        for lambda_2 in quark_helicities:
            for sigma in gluon_polarizations:
                for c1 in quark_colors:
                    for c2 in quark_colors:
                        for a in gluon_colors:
                            quark_gluon_vertex_term_list.append(
                                fixed_quark_gluon_vertex_term(
                                    lambda_1=lambda_1,
                                    lambda_2=lambda_2,
                                    sigma=sigma,
                                    c1=c1,
                                    c2=c2,
                                    cg=a,
                                    mq=mq,
                                    g=g,
                                    K=K,
                                    Kp=Kp,
                                    Nc=Nc,
                                )
                            )
    return quark_gluon_vertex_term_list

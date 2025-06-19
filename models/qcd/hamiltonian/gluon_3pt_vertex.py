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
        nb.float64,
        nb.float64,
        nb.float64,
        nb.int64,
    ),
    ## other options
    fastmath=True,
)
def fixed_gluon_3pt_vertex_term(
    sigma_1: int,
    sigma_2: int,
    sigma_3: int,
    a: int,
    b: int,
    c: int,
    g: float,
    K: float,
    Kp: float,
    Nc: int = 3,
):

    fixed_qns_dict = nb.typed.Dict.empty(
        key_type=nb.types.unicode_type, value_type=nb.types.complex128
    )

    boson_lim = int(K)
    L = 2 * np.pi * K
    Lp = 2 * np.pi * Kp

    boson_longitudinal_q = np.array(
        [i for i in np.arange(-boson_lim, boson_lim + 1.0, 1.0) if i != 0.0],
        dtype=np.float64,
    )
    transverse_q = np.array([i for i in np.arange(-Kp, Kp + 1.0)], dtype=np.float64)

    for q1plus in boson_longitudinal_q:
        for q1trans1 in transverse_q:
            for q1trans2 in transverse_q:
                for q2plus in boson_longitudinal_q:
                    for q2trans1 in transverse_q:
                        for q2trans2 in transverse_q:
                            q1 = np.array(
                                [q1plus, q1trans1, q1trans2], dtype=np.complex128
                            )
                            q2 = np.array(
                                [q2plus, q2trans1, q2trans2], dtype=np.complex128
                            )
                            q3 = -1 * (q2 + q1)  # q3 is assigned by delta

                            if (
                                0 < np.abs(q3[0]) <= K
                                and np.abs(q3[1]) <= Kp
                                and np.abs(q3[2]) <= Kp
                            ):  # make sure k3 < K,Kp

                                Aone = (
                                    heaviside(q1[0]) * pol_vec(p=q1, pol=sigma_1)
                                    + heaviside(-q1[0])
                                    * pol_vec(p=q1, pol=sigma_1).conj()
                                )

                                Atwo = (
                                    heaviside(q2[0]) * pol_vec(p=q2, pol=sigma_2)
                                    + heaviside(-q2[0])
                                    * pol_vec(p=q2, pol=sigma_2).conj()
                                )

                                Athree = (
                                    heaviside(q3[0]) * pol_vec(p=q3, pol=sigma_3)
                                    + heaviside(-q3[0])
                                    * pol_vec(p=q3, pol=sigma_3).conj()
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
                                    * f(a, b, c)
                                    * (1 / ((2 * L) * (Lp) ** 2)) ** 3
                                    / (
                                        2
                                        * np.pi
                                        / L
                                        * (
                                            np.abs(q1[0])
                                            * np.abs(q2[0])
                                            * np.abs(q3[0])
                                        )
                                    )
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
    (nb.float64, nb.float64, nb.float64, nb.int64),
    ## other options
    fastmath=True,
)
def gluon_3pt_vertex_term(g: float, K: float, Kp: float, Nc: int = 3):
    gluon_polarizations = [1, -1]
    gluon_colors = np.arange(1, Nc**2 - 1 + 1, 1)
    gluon_3pt_vertex_term_list = nb.typed.List()

    for sigma_1 in gluon_polarizations:
        for sigma_2 in gluon_polarizations:
            for sigma_3 in gluon_polarizations:
                for a in gluon_colors:
                    for b in gluon_colors:
                        for c in gluon_colors:
                            fabc = f(a, b, c)
                            if fabc != 0:
                                gluon_3pt_vertex_term_list.append(
                                    fixed_gluon_3pt_vertex_term(
                                        sigma_1=sigma_1,
                                        sigma_2=sigma_2,
                                        sigma_3=sigma_3,
                                        a=a,
                                        b=b,
                                        c=c,
                                        g=g,
                                        K=K,
                                        Kp=Kp,
                                        Nc=Nc,
                                    )
                                )

    return gluon_3pt_vertex_term_list

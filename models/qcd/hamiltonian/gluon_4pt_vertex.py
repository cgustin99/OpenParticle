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
def fixed_gluon_4pt_vertex_term(
    sigma_1: int,
    sigma_2: int,
    sigma_3: int,
    sigma_4: int,
    b: int,
    c: int,
    d: int,
    e: int,
    fcoeff: float,
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
                                            Aq1 = (
                                                heaviside(q1[0])
                                                * pol_vec(p=q1, pol=sigma_1)[2:4]
                                                + heaviside(-q1[0])
                                                * pol_vec(p=-q1, pol=sigma_1).conj()[
                                                    2:4
                                                ]
                                            )
                                            Aq3 = (
                                                heaviside(q3[0])
                                                * pol_vec(p=q3, pol=sigma_3)[2:4]
                                                + heaviside(-q3[0])
                                                * pol_vec(p=-q3, pol=sigma_3).conj()[
                                                    2:4
                                                ]
                                            )
                                            Aq2 = (
                                                heaviside(q2[0])
                                                * pol_vec(p=q2, pol=sigma_2)[2:4]
                                                + heaviside(-q2[0])
                                                * pol_vec(p=-q2, pol=sigma_2).conj()[
                                                    2:4
                                                ]
                                            )
                                            Aq4 = (
                                                heaviside(q4[0])
                                                * pol_vec(p=q4, pol=sigma_4)[2:4]
                                                + heaviside(-q4[0])
                                                * pol_vec(p=-q4, pol=sigma_4).conj()[
                                                    2:4
                                                ]
                                            )

                                            coeff = Aq1.dot(Aq3) * Aq2.dot(Aq4)
                                            coeff *= (
                                                g**2
                                                * -1j
                                                * fcoeff
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

                                            if coeff != 0:
                                                particle_operator_string = ""
                                                string_1 = (
                                                    heaviside(-q1[0])
                                                    * heaviside(q2[0])
                                                    * heaviside(q3[0])
                                                    * heaviside(q4[0])
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
                                                                k=q3,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=d,
                                                                pol=sigma_3,
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
                                                                k=q4,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=e,
                                                                pol=sigma_4,
                                                                Nc=Nc,
                                                            )
                                                        )
                                                        + ""
                                                    )
                                                )

                                                string_2 = (
                                                    heaviside(q1[0])
                                                    * heaviside(-q2[0])
                                                    * heaviside(q3[0])
                                                    * heaviside(q4[0])
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
                                                                k=q3,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=d,
                                                                pol=sigma_3,
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
                                                                k=q4,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=e,
                                                                pol=sigma_4,
                                                                Nc=Nc,
                                                            )
                                                        )
                                                        + ""
                                                    )
                                                )

                                                string_3 = (
                                                    heaviside(q1[0])
                                                    * heaviside(q2[0])
                                                    * heaviside(q3[0])
                                                    * heaviside(q4[0])
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
                                                                k=-q3,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=d,
                                                                pol=sigma_3,
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
                                                                k=q4,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=e,
                                                                pol=sigma_4,
                                                                Nc=Nc,
                                                            )
                                                        )
                                                        + ""
                                                    )
                                                )

                                                string_4 = (
                                                    heaviside(q1[0])
                                                    * heaviside(q2[0])
                                                    * heaviside(q3[0])
                                                    * heaviside(-q4[0])
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
                                                                k=q3,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=d,
                                                                pol=sigma_3,
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
                                                                k=-q4,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=e,
                                                                pol=sigma_4,
                                                                Nc=Nc,
                                                            )
                                                        )
                                                        + "^"
                                                    )
                                                )

                                                string_5 = (
                                                    heaviside(-q1[0])
                                                    * heaviside(-q2[0])
                                                    * heaviside(q3[0])
                                                    * heaviside(q4[0])
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
                                                                k=q3,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=d,
                                                                pol=sigma_3,
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
                                                                k=q4,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=e,
                                                                pol=sigma_4,
                                                                Nc=Nc,
                                                            )
                                                        )
                                                        + ""
                                                    )
                                                )

                                                string_6 = (
                                                    heaviside(-q1[0])
                                                    * heaviside(q2[0])
                                                    * heaviside(-q3[0])
                                                    * heaviside(q4[0])
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
                                                                k=-q3,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=d,
                                                                pol=sigma_3,
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
                                                                k=q4,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=e,
                                                                pol=sigma_4,
                                                                Nc=Nc,
                                                            )
                                                        )
                                                        + ""
                                                    )
                                                )

                                                string_7 = (
                                                    heaviside(-q1[0])
                                                    * heaviside(q2[0])
                                                    * heaviside(q3[0])
                                                    * heaviside(-q4[0])
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
                                                                k=q3,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=d,
                                                                pol=sigma_3,
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
                                                                k=-q4,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=e,
                                                                pol=sigma_4,
                                                                Nc=Nc,
                                                            )
                                                        )
                                                        + "^"
                                                    )
                                                )

                                                string_8 = (
                                                    heaviside(q1[0])
                                                    * heaviside(-q2[0])
                                                    * heaviside(-q3[0])
                                                    * heaviside(q4[0])
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
                                                                k=-q3,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=d,
                                                                pol=sigma_3,
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
                                                                k=q4,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=e,
                                                                pol=sigma_4,
                                                                Nc=Nc,
                                                            )
                                                        )
                                                        + ""
                                                    )
                                                )

                                                string_9 = (
                                                    heaviside(q1[0])
                                                    * heaviside(-q2[0])
                                                    * heaviside(q3[0])
                                                    * heaviside(-q4[0])
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
                                                                k=q3,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=d,
                                                                pol=sigma_3,
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
                                                                k=-q4,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=e,
                                                                pol=sigma_4,
                                                                Nc=Nc,
                                                            )
                                                        )
                                                        + "^"
                                                    )
                                                )

                                                string_10 = (
                                                    heaviside(q1[0])
                                                    * heaviside(q2[0])
                                                    * heaviside(-q3[0])
                                                    * heaviside(-q4[0])
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
                                                                k=-q3,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=d,
                                                                pol=sigma_3,
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
                                                                k=-q4,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=e,
                                                                pol=sigma_4,
                                                                Nc=Nc,
                                                            )
                                                        )
                                                        + "^"
                                                    )
                                                )

                                                string_11 = (
                                                    heaviside(-q1[0])
                                                    * heaviside(-q2[0])
                                                    * heaviside(-q3[0])
                                                    * heaviside(q4[0])
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
                                                                k=-q3,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=d,
                                                                pol=sigma_3,
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
                                                                k=q4,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=e,
                                                                pol=sigma_4,
                                                                Nc=Nc,
                                                            )
                                                        )
                                                        + ""
                                                    )
                                                )

                                                string_12 = (
                                                    heaviside(-q1[0])
                                                    * heaviside(-q2[0])
                                                    * heaviside(q3[0])
                                                    * heaviside(-q4[0])
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
                                                                k=q3,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=d,
                                                                pol=sigma_3,
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
                                                                k=-q4,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=e,
                                                                pol=sigma_4,
                                                                Nc=Nc,
                                                            )
                                                        )
                                                        + "^"
                                                    )
                                                )

                                                string_13 = (
                                                    heaviside(-q1[0])
                                                    * heaviside(q2[0])
                                                    * heaviside(-q3[0])
                                                    * heaviside(-q4[0])
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
                                                                k=-q3,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=d,
                                                                pol=sigma_3,
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
                                                                k=-q4,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=e,
                                                                pol=sigma_4,
                                                                Nc=Nc,
                                                            )
                                                        )
                                                        + "^"
                                                    )
                                                )

                                                string_14 = (
                                                    heaviside(q1[0])
                                                    * heaviside(-q2[0])
                                                    * heaviside(-q3[0])
                                                    * heaviside(-q4[0])
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
                                                                k=-q3,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=d,
                                                                pol=sigma_3,
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
                                                                k=-q4,
                                                                K=K,
                                                                Kp=Kp,
                                                                a=e,
                                                                pol=sigma_4,
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
                                                + string_7
                                                + string_8
                                                + string_9
                                                + string_10
                                                + string_11
                                                + string_12
                                                + string_13
                                                + string_14
                                            )

                                            fixed_qns_dict[particle_operator_string] = (
                                                coeff
                                            )
    return fixed_qns_dict


@nb.njit(
    ## output types
    nb.types.ListType(nb.types.DictType(nb.types.unicode_type, nb.types.complex128))
    ##input types
    (nb.float64, nb.float64, nb.float64, nb.int64),
    ## other options
    fastmath=True,
)
def gluon_4pt_vertex_term(g: float, K: float, Kp: float, Nc: int = 3):
    gluon_polarizations = [1, -1]
    gluon_colors = np.arange(1, Nc**2 - 1 + 1, 1)

    gluon_4pt_vertex_term_list = []

    for a in gluon_colors:
        for b in gluon_colors:
            for c in gluon_colors:
                for d in gluon_colors:
                    for e in gluon_colors:
                        for sigma_1 in gluon_polarizations:
                            for sigma_2 in gluon_polarizations:
                                for sigma_3 in gluon_polarizations:
                                    for sigma_4 in gluon_polarizations:
                                        fabc = f(a, b, c)
                                        fade = f(a, d, e)
                                        fcoeff = fabc * fade
                                        if fcoeff != 0:
                                            gluon_4pt_vertex_term_list.append(
                                                fixed_gluon_4pt_vertex_term(
                                                    sigma_1=sigma_1,
                                                    sigma_2=sigma_2,
                                                    sigma_3=sigma_3,
                                                    sigma_4=sigma_4,
                                                    b=b,
                                                    c=c,
                                                    d=d,
                                                    e=e,
                                                    fcoeff=fcoeff,
                                                    g=g,
                                                    K=K,
                                                    Kp=Kp,
                                                    Nc=Nc,
                                                )
                                            )

    return gluon_4pt_vertex_term_list

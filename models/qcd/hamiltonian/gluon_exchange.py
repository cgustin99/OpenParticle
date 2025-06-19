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
def fixed_gluon_4pt_inst_term(
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

                                            coeff = Aq1.dot(Aq2) * Aq3.dot(Aq4)
                                            coeff *= (
                                                g**2
                                                * -1j
                                                * fcoeff
                                                * q2[0]
                                                * q4[0]
                                                / (q3[0] + q4[0]) ** 2
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
def gluon_4pt_inst_term(g: float, K: float, Kp: float, Nc: int = 3):
    gluon_polarizations = [1, -1]
    gluon_colors = np.arange(1, Nc**2 - 1 + 1, 1)

    gluon_4pt_inst_term_list = nb.typed.List()

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
                                            gluon_4pt_inst_term_list.append(
                                                fixed_gluon_4pt_inst_term(
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

    return gluon_4pt_inst_term_list


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
        nb.int64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.int64,
    ),
    ## other options
    fastmath=True,
)
def fixed_quark_4pt_inst_term(
    lambda_1: int,
    lambda_2: int,
    lambda_3: int,
    lambda_4: int,
    a: int,
    c1: int,
    c2: int,
    c3: int,
    c4: int,
    g: float,
    mq: float,
    mg: float,
    K: float,
    Kp: float,
    Nc: int = 3,
):
    fixed_qns_dict = nb.typed.Dict.empty(
        key_type=nb.types.unicode_type, value_type=nb.types.complex128
    )

    fermion_lim = K - ((K - 0.5) % 1)
    L = 2 * np.pi * K
    Lp = 2 * np.pi * Kp

    fermion_longitudinal_q = np.array(
        [i for i in np.arange(-fermion_lim, fermion_lim + 1.0, 1.0) if i != 0.0],
        dtype=np.float64,
    )
    transverse_q = np.array([i for i in np.arange(-Kp, Kp + 1.0)], dtype=np.float64)

    for q1plus in fermion_longitudinal_q:
        for q1trans1 in transverse_q:
            for q1trans2 in transverse_q:
                for q2plus in fermion_longitudinal_q:
                    for q2trans1 in transverse_q:
                        for q2trans2 in transverse_q:
                            for q3plus in fermion_longitudinal_q:
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

                                            psi_1 = heaviside(-q1[0]) * udag(
                                                p=-q1, m=mq, h=lambda_1
                                            ) + heaviside(q1[0]) * vdag(
                                                p=q1, m=mq, h=lambda_1
                                            )
                                            psi_2 = heaviside(q2[0]) * u(
                                                p=q2, m=mq, h=lambda_2
                                            ) + heaviside(-q2[0]) * v(
                                                p=-q2, m=mq, h=lambda_2
                                            )
                                            psi_3 = heaviside(-q3[0]) * udag(
                                                p=-q3, m=mq, h=lambda_3
                                            ) + heaviside(q3[0]) * vdag(
                                                p=q3, m=mq, h=lambda_3
                                            )
                                            psi_4 = heaviside(q4[0]) * u(
                                                p=q4, m=mq, h=lambda_4
                                            ) + heaviside(-q4[0]) * v(
                                                p=-q4, m=mq, h=lambda_4
                                            )

                                            coeff = psi_1.dot(
                                                gamma0.dot(gamma_plus.dot(psi_2))
                                            ) * psi_3.dot(
                                                gamma0.dot(gamma_plus.dot(psi_4))
                                            )

                                            if coeff != 0:
                                                coeff *= (
                                                    g**2
                                                    * T[a - 1][c1 - 1][c2 - 1]
                                                    * T[a - 1][c3 - 1][c4 - 1]
                                                    * 1
                                                    / (
                                                        2
                                                        * np.pi
                                                        / L
                                                        * ((q3[0] + q4[0])) ** 2
                                                        + mg**2
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
                                                            "b"
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
                                                        )
                                                        + heaviside(-q2[0])
                                                        * (
                                                            "d"
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
                                                        )
                                                    )
                                                    + (
                                                        heaviside(-q3[0])
                                                        * (
                                                            "b"
                                                            + str(
                                                                quark_quantum_numbers(
                                                                    k=-q3,
                                                                    K=K,
                                                                    Kp=Kp,
                                                                    c=c3,
                                                                    h=lambda_3,
                                                                    Nc=Nc,
                                                                )
                                                            )
                                                            + "^ "
                                                        )
                                                        + heaviside(q3[0])
                                                        * (
                                                            "d"
                                                            + str(
                                                                quark_quantum_numbers(
                                                                    k=q3,
                                                                    K=K,
                                                                    Kp=Kp,
                                                                    c=c3,
                                                                    h=lambda_3,
                                                                    Nc=Nc,
                                                                )
                                                            )
                                                            + " "
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
                                                            + ""
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
                                                            + "^"
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
    (nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.int64),
    ## other options
    fastmath=True,
)
def quark_4pt_inst_term(
    g: float, mq: float, mg: float, K: float, Kp: float, Nc: int = 3
):
    quark_helicities = [1, -1]
    quark_colors = np.arange(1, Nc + 1, 1)
    gluon_colors = np.arange(1, Nc**2 - 1 + 1, 1)

    quark_4pt_inst_term_list = nb.typed.List()

    for a in gluon_colors:
        for c1 in quark_colors:
            for c2 in quark_colors:
                for c3 in quark_colors:
                    for c4 in quark_colors:
                        for lambda_1 in quark_helicities:
                            for lambda_2 in quark_helicities:
                                for lambda_3 in quark_helicities:
                                    for lambda_4 in quark_helicities:
                                        if (
                                            T[a - 1][c1 - 1][c2 - 1]
                                            * T[a - 1][c3 - 1][c4 - 1]
                                            != 0
                                        ):
                                            quark_4pt_inst_term_list.append(
                                                fixed_quark_4pt_inst_term(
                                                    lambda_1=lambda_1,
                                                    lambda_2=lambda_2,
                                                    lambda_3=lambda_3,
                                                    lambda_4=lambda_4,
                                                    a=a,
                                                    c1=c1,
                                                    c2=c2,
                                                    c3=c3,
                                                    c4=c4,
                                                    g=g,
                                                    mq=mq,
                                                    mg=mg,
                                                    K=K,
                                                    Kp=Kp,
                                                    Nc=Nc,
                                                )
                                            )
    return quark_4pt_inst_term_list


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
def fixed_quark_gluon_inst_term_1(
    lambda_1: int,
    lambda_2: int,
    sigma_3: int,
    sigma_4: int,
    a: int,
    c1: int,
    c2: int,
    d: int,
    e: int,
    g: float,
    mq: float,
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
                for q2plus in fermion_longitudinal_q:
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

                                            psi_1 = heaviside(-q1[0]) * udag(
                                                p=-q1, m=mq, h=lambda_1
                                            ) + heaviside(q1[0]) * vdag(
                                                p=q1, m=mq, h=lambda_1
                                            )
                                            psi_2 = heaviside(q2[0]) * u(
                                                p=q2, m=mq, h=lambda_2
                                            ) + heaviside(-q2[0]) * v(
                                                p=-q2, m=mq, h=lambda_2
                                            )

                                            A3 = heaviside(q3[0]) * pol_vec(
                                                p=q3, pol=sigma_3
                                            ) + heaviside(-q3[0]) * pol_vec(
                                                p=-q3, pol=sigma_3
                                            )
                                            A4 = heaviside(q4[0]) * pol_vec(
                                                p=q4, pol=sigma_4
                                            ) + heaviside(-q4[0]) * pol_vec(
                                                p=-q4, pol=sigma_4
                                            )

                                            coeff = psi_1.dot(
                                                gamma0.dot(gamma_plus.dot(psi_2))
                                            ) * (A3[2:4].dot(A4[2:4]))
                                            if coeff != 0:
                                                coeff *= (
                                                    g**2
                                                    * T[a - 1][c1 - 1][c2 - 1]
                                                    * -1j
                                                    * f(a, d, e)
                                                    * 2
                                                    * np.pi
                                                    * q3[0]
                                                    / L
                                                    * 1
                                                    / (2 * np.pi / L * (q3[0] + q4[0]))
                                                    ** 2
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
                                                            "b"
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
                                                            + "^ "
                                                        )
                                                        + heaviside(-q2[0])
                                                        * (
                                                            "d"
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
                                                            + " "
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
                                                                    a=d,
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
                                                                    a=d,
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
                                                            "a"
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
                                                            + " "
                                                        )
                                                        + heaviside(-q4[0])
                                                        * (
                                                            "a"
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
                                                            + "^ "
                                                        )
                                                    )
                                                )
                                                fixed_qns_dict[
                                                    particle_operator_string
                                                ] = coeff

    return fixed_qns_dict


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
def fixed_quark_gluon_inst_term_2(
    lambda_3: int,
    lambda_4: int,
    sigma_1: int,
    sigma_2: int,
    a: int,
    c3: int,
    c4: int,
    b: int,
    c: int,
    g: float,
    mq: float,
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
                for q2plus in fermion_longitudinal_q:
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

                                            A1 = heaviside(q1[0]) * pol_vec(
                                                p=q1, pol=sigma_1
                                            ) + heaviside(-q1[0]) * pol_vec(
                                                p=-q1, pol=sigma_1
                                            )
                                            A2 = heaviside(q2[0]) * pol_vec(
                                                p=q2, pol=sigma_2
                                            ) + heaviside(-q2[0]) * pol_vec(
                                                p=-q2, pol=sigma_2
                                            )
                                            psi_3 = heaviside(-q3[0]) * udag(
                                                p=-q3, m=mq, h=lambda_3
                                            ) + heaviside(q3[0]) * vdag(
                                                p=q3, m=mq, h=lambda_3
                                            )
                                            psi_4 = heaviside(q4[0]) * u(
                                                p=q4, m=mq, h=lambda_4
                                            ) + heaviside(-q4[0]) * v(
                                                p=-q4, m=mq, h=lambda_4
                                            )

                                            coeff = (A1[2:4].dot(A2[2:4])) * psi_3.dot(
                                                gamma0.dot(gamma_plus.dot(psi_4))
                                            )
                                            if coeff != 0:
                                                coeff *= (
                                                    g**2
                                                    * T[a - 1][c3 - 1][c4 - 1]
                                                    * -1j
                                                    * f(a, b, c)
                                                    * 2
                                                    * np.pi
                                                    * q2[0]
                                                    / L
                                                    * 1
                                                    / (2 * np.pi / L * (q3[0] + q4[0]))
                                                    ** 2
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
                                                        heaviside(q1[0])
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
                                                        )
                                                        + heaviside(-q1[0])
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
                                                                    a=c,
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
                                                                    a=c,
                                                                    pol=sigma_2,
                                                                    Nc=Nc,
                                                                )
                                                            )
                                                            + "^ "
                                                        )
                                                    )
                                                    + (
                                                        heaviside(-q3[0])
                                                        * (
                                                            "b"
                                                            + str(
                                                                quark_quantum_numbers(
                                                                    k=-q3,
                                                                    K=K,
                                                                    Kp=Kp,
                                                                    c=c3,
                                                                    h=lambda_3,
                                                                    Nc=Nc,
                                                                )
                                                            )
                                                            + "^ "
                                                        )
                                                        + heaviside(q3[0])
                                                        * (
                                                            "d"
                                                            + str(
                                                                quark_quantum_numbers(
                                                                    k=q3,
                                                                    K=K,
                                                                    Kp=Kp,
                                                                    c=c3,
                                                                    h=lambda_3,
                                                                    Nc=Nc,
                                                                )
                                                            )
                                                            + " "
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
def quark_gluon_inst_term(g: float, mq: float, K: float, Kp: float, Nc: int = 3):
    quark_helicities = [1, -1]
    gluon_polarizations = [1, -1]
    quark_colors = np.arange(1, Nc + 1, 1)
    gluon_colors = np.arange(1, Nc**2 - 1 + 1, 1)

    quark_gluon_inst_term_list = nb.typed.List()

    for a in gluon_colors:
        for c1 in quark_colors:
            for c2 in quark_colors:
                for d in gluon_colors:
                    for e in gluon_colors:
                        for lambda_1 in quark_helicities:
                            for lambda_2 in quark_helicities:
                                for sigma_1 in gluon_polarizations:
                                    for sigma_2 in gluon_polarizations:

                                        quark_gluon_inst_term_list.append(
                                            fixed_quark_gluon_inst_term_1(
                                                lambda_1=lambda_1,
                                                lambda_2=lambda_2,
                                                sigma_3=sigma_1,
                                                sigma_4=sigma_2,
                                                a=a,
                                                c1=c1,
                                                c2=c2,
                                                d=d,
                                                e=e,
                                                g=g,
                                                mq=mq,
                                                K=K,
                                                Kp=Kp,
                                                Nc=Nc,
                                            )
                                        )
                                        quark_gluon_inst_term_list.append(
                                            fixed_quark_gluon_inst_term_2(
                                                lambda_3=lambda_1,
                                                lambda_4=lambda_2,
                                                sigma_1=sigma_1,
                                                sigma_2=sigma_2,
                                                a=a,
                                                c3=c1,
                                                c4=c2,
                                                b=d,
                                                c=e,
                                                g=g,
                                                mq=mq,
                                                K=K,
                                                Kp=Kp,
                                                Nc=Nc,
                                            )
                                        )
    return quark_gluon_inst_term_list

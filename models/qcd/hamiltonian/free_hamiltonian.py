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
    (nb.int64, nb.int64, nb.float64, nb.float64, nb.float64, nb.int64),
    ## other options
    fastmath=True,
)
def fixed_free_quark_hamiltonian(
    color: int, helicity: int, K: float, Kp: float, mq: float, Nc: int = 3
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

    for i in fermion_longitudinal_q:
        for j in transverse_q:
            for k in transverse_q:
                q = np.array([i, j, k], dtype=np.complex128)
                psi_dag = heaviside(q[0]) * udag(p=q, m=mq, h=helicity) + heaviside(
                    -q[0]
                ) * vdag(p=-q, m=mq, h=helicity)
                psi = heaviside(q[0]) * u(p=q, m=mq, h=helicity) + heaviside(-q[0]) * v(
                    p=-q, m=mq, h=helicity
                )
                coeff = psi_dag.dot(gamma0.dot(gamma_plus.dot(psi)))
                coeff *= (
                    0.5
                    * (1 / ((2 * L) * (Lp) ** 2))
                    * ((q[1:3].dot(q[1:3]) + mq**2) / q[0])
                )
                particle_operator_string = (
                    heaviside(q[0])
                    * (
                        "b"
                        + str(
                            quark_quantum_numbers(
                                k=q, K=K, Kp=Kp, c=color, h=helicity, Nc=Nc
                            )
                        )
                        + "^ "
                    )
                    + heaviside(-q[0])
                    * (
                        "d"
                        + str(
                            quark_quantum_numbers(
                                k=-q, K=K, Kp=Kp, c=color, h=helicity, Nc=Nc
                            )
                        )
                        + " "
                    )
                ) + (
                    heaviside(q[0])
                    * (
                        "b"
                        + str(
                            quark_quantum_numbers(
                                k=q, K=K, Kp=Kp, c=color, h=helicity, Nc=Nc
                            )
                        )
                        + " "
                    )
                    + heaviside(-q[0])
                    * (
                        "d"
                        + str(
                            quark_quantum_numbers(
                                k=-q, K=K, Kp=Kp, c=color, h=helicity, Nc=Nc
                            )
                        )
                        + "^ "
                    )
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
def free_quark_hamiltonian(K: float, Kp: float, mq: float, Nc: int = 3):
    quark_helicities = [1, -1]
    quark_colors = np.arange(1, Nc + 1, 1)

    free_quark_term_list = nb.typed.List()
    for color in quark_colors:
        for helicity in quark_helicities:
            free_quark_term_list.append(
                fixed_free_quark_hamiltonian(
                    color=color, helicity=helicity, K=K, Kp=Kp, mq=mq, Nc=Nc
                )
            )
    return free_quark_term_list


@nb.njit(
    ## output types
    nb.types.DictType(nb.types.string, nb.types.complex128)
    ##input types
    (nb.int64, nb.int64, nb.float64, nb.float64, nb.float64, nb.int64),
    ## other options
    fastmath=True,
)
def fixed_free_gluon_hamiltonian(
    color: int, polarization: int, K: float, Kp: float, mg: float, Nc: int = 3
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

    for i in boson_longitudinal_q:
        for j in transverse_q:
            for k in transverse_q:
                q = np.array([i, j, k], dtype=np.complex128)

                Aplus = (
                    heaviside(q[0]) * pol_vec(p=q, pol=polarization)
                    + heaviside(-q[0]) * pol_vec(p=-q, pol=polarization).conj()
                )
                Aminus = (
                    heaviside(-q[0]) * pol_vec(p=-q, pol=polarization)
                    + heaviside(q[0]) * pol_vec(p=q, pol=polarization).conj()
                )
                coeff = -1 * Aminus[2] * Aplus[2] - 1 * Aminus[3] * Aplus[3]
                coeff *= (
                    0.5 * (1 / ((2 * L) * (Lp) ** 2)) * ((q[1:3].dot(q[1:3]) + mg**2))
                )
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


@nb.njit(
    ## output types
    nb.types.ListType(nb.types.DictType(nb.types.unicode_type, nb.types.complex128))
    ##input types
    (nb.float64, nb.float64, nb.float64, nb.int64),
    ## other options
    fastmath=True,
)
def free_gluon_hamiltonian(K, Kp, mg, Nc=3):
    gluon_polarizations = [1, -1]
    gluon_colors = np.arange(1, Nc**2 - 1 + 1, 1)

    free_gluon_term_list = nb.typed.List()
    for color in gluon_colors:
        for polarization in gluon_polarizations:
            free_gluon_term_list.append(
                fixed_free_gluon_hamiltonian(
                    color=color, polarization=polarization, K=K, Kp=Kp, mg=mg, Nc=Nc
                )
            )
    return free_gluon_term_list

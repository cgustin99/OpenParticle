import openparticle as op
from openparticle.full_dlcq import *
from color_algebra import T, f
from effective_mass import *
import numpy as np

# import numba as nb

Nc = 3


@nb.njit(
    ## output types
    nb.complex128[:, :, :, :, :]
    ##input types
    (nb.float64, nb.float64, nb.float64),
    ## other options
    fastmath=True,
)
def free_quark_hamiltonian_tensor(K: float, Kp: float, mq: float):
    quark_helicities = np.array([1, -1], dtype=np.int8)
    quark_colors = np.arange(1, Nc + 1, 1)
    fermion_lim = K - ((K - 0.5) % 1)
    fermion_longitudinal_q = np.array(
        [i for i in np.arange(-fermion_lim, fermion_lim + 1.0, 1.0) if i != 0.0],
        dtype=np.float64,
    )
    transverse_q = np.array([i for i in np.arange(-Kp, Kp + 1.0)], dtype=np.float64)

    dim = (
        len(fermion_longitudinal_q),
        len(transverse_q),
        len(transverse_q),
        len(quark_colors),
        len(quark_helicities),
    )
    free_quark_tensor = np.zeros(dim, dtype=np.complex128)

    for color in quark_colors:
        for helicity in quark_helicities:
            for qplus in fermion_longitudinal_q:
                for qperp1 in transverse_q:
                    for qperp2 in transverse_q:
                        q = np.array([qplus, qperp1, qperp2], dtype=np.complex128)
                        coeff = (mq**2 + q[1:3].dot(q[1:3])) / q[0]

                        color_index = color - 1
                        helicity_index = np.where(helicity == quark_helicities)[0][0]
                        qplus_index = np.where(fermion_longitudinal_q == qplus)[0][0]
                        qperp1_index = np.where(transverse_q == qperp1)[0][0]
                        qperp2_index = np.where(transverse_q == qperp2)[0][0]

                        free_quark_tensor[
                            qplus_index,
                            qperp1_index,
                            qperp2_index,
                            color_index,
                            helicity_index,
                        ] = coeff

    return free_quark_tensor


@nb.njit(
    ## output types
    nb.complex128[:, :, :, :, :]
    ##input types
    (nb.float64, nb.float64, nb.float64),
    ## other options
    fastmath=True,
)
def free_gluon_hamiltonian_tensor(K: float, Kp: float, mg: float):
    gluon_polarizations = [1, -1]
    gluon_colors = np.arange(1, Nc**2 - 1 + 1, 1)
    boson_lim = int(K)
    boson_longitudinal_q = np.arange(1, boson_lim + 1, 1, dtype=np.float64)
    transverse_q = np.array([i for i in np.arange(-Kp, Kp + 1.0)], dtype=np.float64)

    dim = (
        len(boson_longitudinal_q),
        len(transverse_q),
        len(transverse_q),
        len(gluon_colors),
        len(gluon_polarizations),
    )
    free_gluon_tensor = np.zeros(dim, dtype=np.complex128)

    for color in gluon_colors:
        for polarization in gluon_polarizations:
            for qplus in boson_longitudinal_q:
                for qperp1 in transverse_q:
                    for qperp2 in transverse_q:
                        q = np.array([qplus, qperp1, qperp2], dtype=np.complex128)
                        coeff = (mg**2 + q[1:3].dot(q[1:3])) / q[0]

                        color_index = color - 1
                        polarization_index = polarization - 1
                        qplus_index = np.where(boson_longitudinal_q == qplus)[0][0]
                        qperp1_index = np.where(transverse_q == qperp1)[0][0]
                        qperp2_index = np.where(transverse_q == qperp2)[0][0]

                        free_gluon_tensor[
                            qplus_index,
                            qperp1_index,
                            qperp2_index,
                            color_index,
                            polarization_index,
                        ] = coeff

    return free_gluon_tensor


# @nb.njit(
#     ## output types
#     nb.complex128[:, :, :, :, :]
#     ##input types
#     (nb.float64, nb.float64, nb.float64, nb.float64, nb.float64),
#     ## other options
#     fastmath=True,
# )
def effective_free_quark_hamiltonian_tensor(
    s: float, K: float, Kp: float, mq: float, mg: float
):
    quark_helicities = np.array([1, -1], dtype=np.int8)
    quark_colors = np.arange(1, Nc + 1, 1)
    fermion_lim = K - ((K - 0.5) % 1)
    fermion_longitudinal_q = np.array(
        [i for i in np.arange(-fermion_lim, fermion_lim + 1.0, 1.0) if i != 0.0],
        dtype=np.float64,
    )
    transverse_q = np.array([i for i in np.arange(-Kp, Kp + 1.0)], dtype=np.float64)

    dim = (
        len(fermion_longitudinal_q),
        len(transverse_q),
        len(transverse_q),
        len(quark_colors),
        len(quark_helicities),
    )
    free_quark_tensor = np.zeros(dim, dtype=np.complex128)

    for color in quark_colors:
        for helicity in quark_helicities:
            for qplus in fermion_longitudinal_q:
                for qperp1 in transverse_q:
                    for qperp2 in transverse_q:
                        q = np.array([qplus, qperp1, qperp2], dtype=np.complex128)
                        coeff = (
                            effective_quark_mass_sq(s=s, q=qplus, mq=mq, mg=mg)
                            + q[1:3].dot(q[1:3])
                        ) / q[0]

                        color_index = color - 1
                        helicity_index = np.where(helicity == quark_helicities)[0][0]
                        qplus_index = np.where(fermion_longitudinal_q == qplus)[0][0]
                        qperp1_index = np.where(transverse_q == qperp1)[0][0]
                        qperp2_index = np.where(transverse_q == qperp2)[0][0]

                        free_quark_tensor[
                            qplus_index,
                            qperp1_index,
                            qperp2_index,
                            color_index,
                            helicity_index,
                        ] = coeff

    return free_quark_tensor


# @nb.njit(
#     ## output types
#     nb.complex128[:, :, :, :, :]
#     ##input types
#     (nb.float64, nb.float64, nb.float64, nb.float64, nb.float64),
#     ## other options
#     fastmath=True,
# )
def effective_free_gluon_hamiltonian_tensor(
    s: float, K: float, Kp: float, mg: float, mq: float
):
    gluon_polarizations = [1, -1]
    gluon_colors = np.arange(1, Nc**2 - 1 + 1, 1)
    boson_lim = int(K)
    boson_longitudinal_q = np.arange(1, boson_lim + 1, 1, dtype=np.float64)
    transverse_q = np.array([i for i in np.arange(-Kp, Kp + 1.0)], dtype=np.float64)

    dim = (
        len(boson_longitudinal_q),
        len(transverse_q),
        len(transverse_q),
        len(gluon_colors),
        len(gluon_polarizations),
    )
    free_gluon_tensor = np.zeros(dim, dtype=np.complex128)

    for color in gluon_colors:
        for polarization in gluon_polarizations:
            for qplus in boson_longitudinal_q:
                for qperp1 in transverse_q:
                    for qperp2 in transverse_q:
                        q = np.array([qplus, qperp1, qperp2], dtype=np.complex128)
                        coeff = (
                            effective_gluon_mass_sq(s=s, q=qplus, mq=mq, mg=mg)
                            + q[1:3].dot(q[1:3])
                        ) / q[0]

                        color_index = color - 1
                        polarization_index = polarization - 1
                        qplus_index = np.where(boson_longitudinal_q == qplus)[0][0]
                        qperp1_index = np.where(transverse_q == qperp1)[0][0]
                        qperp2_index = np.where(transverse_q == qperp2)[0][0]

                        free_gluon_tensor[
                            qplus_index,
                            qperp1_index,
                            qperp2_index,
                            color_index,
                            polarization_index,
                        ] = coeff

    return free_gluon_tensor

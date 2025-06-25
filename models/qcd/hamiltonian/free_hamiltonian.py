import openparticle as op
from openparticle.full_dlcq import *
from color_algebra import T, f
import numpy as np
import numba as nb


@nb.njit(
    ## output types
    nb.complex128[:, :, :, :, :]
    ##input types
    (nb.float64, nb.float64, nb.float64, nb.int64),
    ## other options
    fastmath=True,
)
def free_quark_hamiltonian_tensor(K: float, Kp: float, mq: float, Nc: int = 3):
    quark_helicities = [1, -1]
    quark_colors = np.arange(1, Nc + 1, 1)
    fermion_lim = K - ((K - 0.5) % 1)
    fermion_longitudinal_q = np.array(
        [i for i in np.arange(-fermion_lim, fermion_lim + 1.0, 1.0) if i != 0.0],
        dtype=np.float64,
    )
    transverse_q = np.array([i for i in np.arange(-Kp, Kp + 1.0)], dtype=np.float64)

    dim = (
        len(quark_colors),
        len(quark_helicities),
        len(fermion_longitudinal_q),
        len(transverse_q),
        len(transverse_q),
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
                        helicity_index = helicity - 1
                        qplus_index = np.where(fermion_longitudinal_q == qplus)[0][0]
                        qperp1_index = np.where(transverse_q == qperp1)[0][0]
                        qperp2_index = np.where(transverse_q == qperp2)[0][0]
                        free_quark_tensor[
                            color_index,
                            helicity_index,
                            qplus_index,
                            qperp1_index,
                            qperp2_index,
                        ] = coeff

    return free_quark_tensor


@nb.njit(
    ## output types
    nb.complex128[:, :, :, :, :]
    ##input types
    (nb.float64, nb.float64, nb.float64, nb.int64),
    ## other options
    fastmath=True,
)
def free_gluon_hamiltonian_tensor(K: float, Kp: float, mg: float, Nc: int = 3):
    gluon_polarizations = [1, -1]
    gluon_colors = np.arange(1, Nc**2 - 1 + 1, 1)
    boson_lim = int(K)
    boson_longitudinal_q = np.arange(1, boson_lim + 1, 1, dtype=np.float64)
    transverse_q = np.array([i for i in np.arange(-Kp, Kp + 1.0)], dtype=np.float64)

    dim = (
        len(gluon_colors),
        len(gluon_polarizations),
        len(boson_longitudinal_q),
        len(transverse_q),
        len(transverse_q),
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
                            color_index,
                            polarization_index,
                            qplus_index,
                            qperp1_index,
                            qperp2_index,
                        ] = coeff

    return free_gluon_tensor

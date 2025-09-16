import numpy as np
import numba as nb


@nb.njit(
    nb.complex128(
        nb.complex128,  # Q = q1minus + q2minus + q3minus
        nb.complex128,  # t
    ),
    fastmath=True,
)
def rgpep_factor_first_order(Q, t):
    # R -> f(Q, t) = np.exp(-(Q^-)**2 * t)
    # Q = q1minus + q2minus + q3minus
    return np.exp(-1 * Q**2 * t)


@nb.njit(nb.complex128(nb.complex128, nb.complex128, nb.complex128), fastmath=True)
def rgpep_factor_second_order(Qm, Qpm, t):
    # R(Q, t) = 1/2 * (1/Q^- - 1/Q'^-)*(f(Q^-, t)f(Q'^-, t) - f(Q^- + Q'^-, t))
    return (
        0.5
        * (1 / Qm - 1 / Qpm)
        * (
            rgpep_factor_first_order(Qm, t) * rgpep_factor_first_order(Qpm, t)
            - rgpep_factor_first_order(Qm + Qpm, t)
        )
    )

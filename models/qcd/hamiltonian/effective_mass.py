from scipy.integrate import dblquad
import numpy as np
import numba as nb


# Calculate m_i^2(s) for i = quark, antiquark, gluon


@nb.njit(
    ## output types
    nb.float64,
    ##input types
    (nb.float64, nb.float64, nb.float64, nb.float64),
    ## other options
    fastmath=True,
)
def effective_quark_mass_sq(s, q, mq, mg):

    def integrand(x, Msq, s, q, mq, mg):
        return (
            (
                (2 * x) / (1 - x) * (x * Msq + mg**2)
                + 2
                * ((Msq - mg**2 - mq**2) / (1 - x) - (2 * mg**2 * x) / ((1 - x) ** 2))
                - 4 * mq**2
            )
            / (Msq - mq**2)
            * np.exp(-2 * s / (q**2) * (Msq - mq**2) ** 2)
        )

    return (
        mq**2
        + dblquad(
            integrand,
            0,
            1,
            lambda x: mq**2 / x + mg**2 / (1 - x),
            np.inf,
            args=(s, q, mq, mg),
        )[0]
    )


@nb.njit(
    ## output types
    nb.float64,
    ##input types
    (nb.float64, nb.float64, nb.float64, nb.float64),
    ## other options
    fastmath=True,
)
def effective_gluon_mass_sq(s, q, mq, mg):
    Msq_min = (mq + mg) ** 2

    def integrand_quark_loop(x, Msq, s, q, mq, mg):
        return (
            (Msq * (2 * x**2 - 2 * x + 1) - mq**2)
            / (Msq - mq**2)
            * np.exp(-2 * s / (q**2) * (Msq - mq**2) ** 2)
        )

    def integrand_gluon_loop(x, Msq, s, q, mq, mg):
        return (
            (Msq * x * (1 - x) - mg**2)
            / (x * (1 - x))
            * (1 + 1 / (x**2) + 1 / ((1 - x) ** 2))
            * np.exp(-2 * s / (q**2) * (Msq - mq**2) ** 2)
        )

    return (
        mg**2
        + dblquad(
            integrand_quark_loop,
            0,
            1,
            lambda x: mq**2 / x + mg**2 / (1 - x),
            np.inf,
            args=(s, q, mq, mg),
        )[0]
        + dblquad(
            integrand_gluon_loop,
            0,
            1,
            lambda x: mg**2 / (x * (1 - x)),
            np.inf,
            args=(s, q, mq, mg),
        )[0]
    )

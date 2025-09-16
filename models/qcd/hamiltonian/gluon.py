import numpy as np
import numba as nb


g_upper_indices = np.array(
    [[0, 0.5, 0, 0], [0.5, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
)  # g^{\mu \nu}
g_lower_indices = np.array(
    [[0, 2, 0, 0], [2, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=np.float64
)  # g_{\mu \nu}


n = np.array([0, 2, 0, 0], dtype=np.float64)


@nb.njit(
    nb.complex128(
        nb.complex128[:], nb.float64, nb.int64, nb.int64, nb.types.unicode_type
    ),
    fastmath=True,
)
def d(q, mg, mu, nu, indices: str = "upper"):
    # Indices 0, 1, 2, 3 correspond to +, -, perp1, perp2
    d_tensor = (
        -1 * g_upper_indices[mu, nu]
        + 1 / q[0] * (q[mu] * n[nu] + q[nu] * n[mu])
        + mg**2 / q[0] ** 2 * (n[mu] * n[nu])
    )
    if indices == "upper":
        return d_tensor
    elif indices == "lower":
        return g_lower_indices[mu, nu] * d_tensor
    else:
        raise ValueError("indices must be 'upper' or 'lower'")

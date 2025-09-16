import numpy as np
import numba as nb

# Gell-Mann Matrices
T = 0.5 * np.array(
    [
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
        np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]]),
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
        np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
        np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]]),
        np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
        np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]]),
        1
        / np.sqrt(3)
        * np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, -2],
            ]
        ),
    ]
)


@nb.njit(nb.float64(nb.int8, nb.int8, nb.int8))
# structure constants
def f(a, b, c):
    structure_constants_array = np.zeros((9, 9, 9), dtype=np.float64)
    structure_constants_array[0, 1, 2] = 1  # "123"
    structure_constants_array[0, 3, 6] = 0.5  # "147"
    structure_constants_array[0, 4, 5] = -0.5  # "156"
    structure_constants_array[1, 3, 5] = 0.5  # "246"
    structure_constants_array[1, 4, 6] = 0.5  # "257"
    structure_constants_array[2, 3, 4] = 0.5  # "345"
    structure_constants_array[2, 5, 6] = -0.5  # "367"
    structure_constants_array[3, 4, 7] = np.sqrt(3) / 2  # "458"
    structure_constants_array[5, 6, 7] = np.sqrt(3) / 2  # "678"

    return structure_constants_array[a - 1][b - 1][c - 1]

import numpy as np

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


# structure constants
def f(a, b, c):
    structure_constants = {
        "123": 1,
        "147": 1 / 2,
        "156": -1 / 2,
        "246": 1 / 2,
        "257": 1 / 2,
        "345": 1 / 2,
        "367": -1 / 2,
        "458": np.sqrt(3) / 2,
        "678": np.sqrt(3) / 2,
    }

    return structure_constants.get(str(a + 1) + str(b + 1) + str(c + 1), 0)

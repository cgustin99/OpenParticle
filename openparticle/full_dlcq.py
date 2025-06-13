import openparticle as op
import numpy as np
import warnings

warnings.filterwarnings("ignore")

sigma1 = np.array([[0, 1], [1, 0]])
sigma2 = np.array([[0, -1j], [1j, 0]])
I2 = np.eye(2)
zeros = np.zeros([2, 2])

gamma0 = np.block([[zeros, I2], [I2, zeros]])
gamma3 = np.block([[zeros, -I2], [I2, zeros]])
gamma1 = np.block([[-1j * sigma2, zeros], [zeros, 1j * sigma2]])
gamma2 = np.block([[-1j * sigma1, zeros], [zeros, 1j * sigma1]])

gamma_plus = gamma0 + gamma3


def u(p, m, h):
    # p: [p^+, p^1, p^2]
    eta = np.array([h == 1, h == -1], dtype=int).reshape([-1, 1])
    component_1 = p[0] * np.eye(2).dot(eta)
    component_2 = (1j * p[2] * sigma1 - 1j * p[1] * sigma2 + m * np.eye(2)).dot(eta)
    return 1 / np.sqrt(np.abs(p[0])) * np.concatenate([component_1, component_2])


def udag(p, m, h):
    return u(p, m, h).conj().T


def v(p, m, h):
    eta = np.array([h == -1, h == 1], dtype=int)
    component_1 = -p[0] * np.eye(2).dot(eta)
    component_2 = (-1j * p[2] * sigma1 + 1j * p[1] * sigma2 + m * np.eye(2)).dot(eta)
    return (
        1
        / np.sqrt(np.abs(p[0]))
        * np.concatenate([component_1, component_2]).reshape([-1, 1])
    )


def vdag(p, m, h):
    return v(p, m, h).conj().T


def pol_vec(p, pol):
    if isinstance(p, list):
        p = np.array(p)
    eps_perp = -1 / np.sqrt(2) * np.array([pol, 1j]).reshape([-1, 1])
    return np.array(
        [0, (2 * p[1:].T.dot(eps_perp) / p[0])[0], eps_perp[0][0], eps_perp[1][0]]
    )


def quark_quantum_numbers(k, kp, K, Kp, c, h, Nc=3):
    # k ∈ [1/2, K]
    # kp ∈ [-Kp, Kp]^2 / 0
    # quark_color: c ∈ {1, 2, 3}
    # spin_proj: h ∈ {-1, +1}

    k_index = k - 1 / 2
    # kp_indices = [kpi + Kp for kpi in kp]
    kp_indices = [[i for i in np.arange(-Kp, Kp + 1)].index(i) for i in kp]
    color_index = c - 1

    if h == 1:
        h_index = 0
    else:
        h_index = 1

    K_size = K + ((K) % 1 - 0.5)

    Kp_size = 2 * Kp + 1
    index = int(
        (
            ((k_index * Kp_size + kp_indices[0]) * Kp_size + kp_indices[1]) * Nc
            + color_index
        )
        * 2
        + h_index
    )
    return index


def decode_quark_quantum_numbers(index, K, Kp, Nc=3):
    helicity_index = index % 2
    helicity = 1 if helicity_index == 0 else -1
    index //= 2

    color_index = index % Nc
    color = color_index + 1
    index //= Nc

    kp1_index = index % (2 * Kp + 1)
    index //= 2 * Kp + 1
    kp0_index = index % (2 * Kp + 1)
    index //= 2 * Kp + 1

    kp0, kp1 = kp0_index - Kp, kp1_index - Kp

    k_index = int(index % (K + ((K) % 1 - 0.5)))
    k = k_index + 1 / 2

    return k, [kp0, kp1], color, helicity


def gluon_quantum_numbers(k, kp, K, Kp, a, pol, Nc=3):
    # k ∈ [1, K]
    # kp ∈ [-Kp, Kp]^2
    # gluon_color: a ∈ {1, 2, 3, 4, 5, 6, 7, 8}
    # spin_proj: pol ∈ {-1, +1}

    k_index = k - 1
    # kp_indices = [kpi + Kp for kpi in kp]
    kp_indices = [[i for i in np.arange(-Kp, Kp + 1)].index(i) for i in kp]
    color_index = a - 1

    if pol == 1:
        polarization_index = 0
    else:
        polarization_index = 1

    K_size = K - ((K) % 1)
    Kp_size = 2 * Kp + 1
    index = int(
        (
            ((k_index * Kp_size + kp_indices[0]) * Kp_size + kp_indices[1])
            * (Nc**2 - 1)
            + color_index
        )
        * 2
        + polarization_index
    )
    return index


def decode_gluon_quantum_numbers(index, K, Kp, Nc=3):
    polarization_index = index % 2
    polarization = 1 if polarization_index == 0 else +1
    index //= 2

    color_index = index % (Nc**2 - 1)
    color = color_index + 1
    index //= Nc**2 - 1

    kp1_index = index % (2 * Kp + 1)
    index //= 2 * Kp + 1
    kp0_index = index % (2 * Kp + 1)
    index //= 2 * Kp + 1
    kp0, kp1 = kp0_index - Kp, kp1_index - Kp

    k_index = int(index % (K - (K % 1)))
    k = k_index + 1

    print(polarization, color, kp0, kp1, k)


class QuarkField:

    def __init__(self, m, k, kp, K, Kp, c, h):
        L = 2 * np.pi * K
        Lp = 2 * np.pi * Kp
        p = np.array(
            [2 * np.pi * k / L, 2 * np.pi * kp[0] / Lp, 2 * np.pi * kp[1] / Lp]
        )
        self.psi = (
            (2 * L)
            / np.sqrt(4 * np.pi * np.abs(k))
            * (
                np.heaviside(k, 0)
                * u(p=p, m=m, h=h)
                * op.ParticleOperator(
                    "b" + str(quark_quantum_numbers(k=k, kp=kp, K=K, Kp=Kp, c=c, h=h))
                )
                + np.heaviside(-k, 0)
                * v(p=-1 * p, m=m, h=h)
                * op.ParticleOperator(
                    "d"
                    + str(
                        quark_quantum_numbers(
                            k=-1 * k, kp=np.array(kp), K=K, Kp=Kp, c=c, h=h
                        )
                    )
                    + "^"
                )
            )
        )

        self.psi_dag = (
            (2 * L)
            / np.sqrt(4 * np.pi * np.abs(k))
            * (
                np.heaviside(k, 0)
                * ubar(p=p, m=m, h=h)
                * op.ParticleOperator(
                    "b"
                    + str(quark_quantum_numbers(k=k, kp=kp, K=K, Kp=Kp, c=c, h=h))
                    + "^"
                )
                + np.heaviside(-k, 0)
                * vbar(p=-1 * p, m=m, h=h)
                * op.ParticleOperator(
                    "d"
                    + str(
                        quark_quantum_numbers(
                            k=-k, kp=np.array(kp), K=K, Kp=Kp, c=c, h=h
                        )
                    )
                )
            )
        )


class GluonField:

    def __init__(self, mg, k, kp, K, Kp, a, pol):
        L = 2 * np.pi * K
        Lp = 2 * np.pi * Kp
        p = np.array(
            [2 * np.pi * k / L, 2 * np.pi * kp[0] / Lp, 2 * np.pi * kp[1] / Lp]
        )

        self.A = (
            (2 * L)
            / np.sqrt(4 * np.pi * np.abs(k))
            * (
                np.heaviside(k, 0)
                * pol_vec(p=p, pol=pol)
                * op.ParticleOperator(
                    "a"
                    + str(gluon_quantum_numbers(k=k, kp=kp, K=K, Kp=Kp, a=a, pol=pol))
                )
                + np.heaviside(-k, 0)
                * pol_vec(p=-1 * p, pol=pol).conj()
                * op.ParticleOperator(
                    "a"
                    + str(gluon_quantum_numbers(k=-k, kp=kp, K=K, Kp=Kp, a=a, pol=pol))
                    + "^"
                )
            )
        )

import numpy as np


# gamma matrices
gamma0 = np.array([[0, 1], [1, 0]])
gamma1 = np.array([[0, -1], [1, 0]])
gammap = gamma0 + gamma1
Lambdap = gamma0.dot(gammap)


# discretized lightfront momentum
def p(k, L=1):
    return 2 * np.pi * k / L


# 1 + 1D lightcone spinors
def u(k, m=1, L=1):
    return 1 / np.sqrt(np.abs(p(k, L))) * np.array([p(k, L), m]).reshape([-1, 1])


def v(k, m=1, L=1):
    return 1 / np.sqrt(np.abs(p(k, L))) * np.array([-p(k, L), m]).reshape([-1, 1])


def ubar(k, m=1, L=1):
    return (u(k, m, L).reshape([1, -1])).dot(
        gamma0
    )  # u(p) is real so dagger -> transpose


def vbar(k, m=1, L=1):
    return (
        v(k, m, L).reshape([1, -1]).dot(gamma0)
    )  # v(p) is real so dagger -> transpose

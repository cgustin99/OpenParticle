import openparticle as op
from openparticle.full_dlcq import *
from models.QCD.hamiltonian.gell_mann import T
import numpy as np


def qcd_hamiltonian(K, Kp, mq, mg, g, Nc=3):
    return (
        free_quark_hamiltonian(K, Kp, Nc, mq)
        + free_gluon_hamiltonian(K, Kp, Nc, mg)
        + quark_gluon_vertex(K, Kp, Nc, mq, mg, g)
        + three_pt_gluon_vertex(K, Kp, Nc, mq, mg, g)
        + four_pt_gluon_vertex(K, Kp, Nc, mq, mg, g)
        + instantaneous_gluon(K, Kp, Nc, mq, mg, g)
        + instantaneous_quark(K, Kp, Nc, mq, mg, g)
    )


def free_quark_hamiltonian(K, Kp, mq, Nc=3):
    ham = op.ParticleOperator()
    L = 2 * np.pi * K
    Lp = 2 * np.pi * Kp
    lim = K - ((K - 0.5) % 1)
    for c in np.arange(1, Nc + 1, 1):
        for h in [1, -1]:
            for k in np.arange(-lim, lim + 1, 1):
                for kp1 in range(-Kp, Kp):
                    for kp2 in range(-Kp, Kp):
                        qperp_sq = (2 * np.pi / Lp) ** 2 * (kp1**2 + kp2**2)
                        qplus = 2 * np.pi * k / L
                        q_minus = (mq**2 + qperp_sq) / qplus
                        ham += 0.5 * (
                            q_minus
                            * (
                                QuarkField(
                                    m=mq, k=k, K=K, Kp=Kp, kp=[kp1, kp2], c=c, h=h
                                ).psi_dag.dot(
                                    gamma0.dot(
                                        gamma_plus.dot(
                                            QuarkField(
                                                m=mq,
                                                k=k,
                                                K=K,
                                                Kp=Kp,
                                                kp=[kp1, kp2],
                                                c=c,
                                                h=h,
                                            ).psi
                                        )
                                    )
                                )
                            )[0][0]
                        )
    ham = ham.normal_order()
    ham.remove_identity()
    ham = 1 / ((2 * L) * (Lp) ** 2) * ham

    return ham


def free_gluon_hamiltonian(K, Kp, mg, Nc=3):
    ham = op.ParticleOperator()
    L = 2 * np.pi * K
    Lp = 2 * np.pi * Kp
    lim = int(K)
    for a in np.arange(1, Nc**2 - 1 + 1, 1):
        for pol in [1, -1]:
            for k in [i for i in range(-lim, lim + 1) if i != 0]:
                for kp1 in range(-Kp, Kp):
                    for kp2 in range(-Kp, Kp):
                        qperp_sq = (2 * np.pi / Lp) ** 2 * (kp1**2 + kp2**2)
                        ham += (mg**2 + qperp_sq) * (
                            VectorField(
                                mg=mg,
                                k=-k,
                                K=K,
                                Kp=Kp,
                                kp=np.array([kp1, kp2]),
                                a=a,
                                pol=pol,
                            ).A.dot(
                                VectorField(
                                    mg=mg, k=k, K=K, Kp=Kp, kp=[kp1, kp2], a=a, pol=pol
                                ).A
                            )
                        )
    ham = ham.normal_order()
    ham.remove_identity()
    ham = 1 / ((2 * L) * (Lp) ** 2) * ham
    return ham


def quark_gluon_vertex(K, Kp, Nc, mq, mg, g):
    ham = op.ParticleOperator()
    L = 2 * np.pi * K
    Lp = 2 * np.pi * Kp
    fermion_lim = K - ((K - 0.5) % 1)
    boson_lim = int(K)
    for c1 in np.arange(1, Nc + 1, 1):
        for c2 in np.arange(1, Nc + 1, 1):
            for a in np.arange(1, Nc**2 - 1 + 1, 1):
                color_factor = T[a - 1][c1 - 1][c2 - 1]
                for h1 in [1, -1]:
                    for h2 in [1, -1]:
                        for pol in [1, -1]:
                            for k1 in np.arange(-fermion_lim, fermion_lim + 1, 1):
                                for kp1x in range(-Kp, Kp):
                                    for kp1y in range(-Kp, Kp):
                                        psi_dag = QuarkField(
                                            m=mq,
                                            k=-k1,
                                            kp=-1 * np.array([kp1x, kp1y]),
                                            K=K,
                                            Kp=Kp,
                                            c=c1,
                                            h=h1,
                                        ).psi_dag
                                        for k3 in np.arange(
                                            -fermion_lim, fermion_lim + 1, 1
                                        ):
                                            for kp3x in range(-Kp, Kp):
                                                for kp3y in range(-Kp, Kp):
                                                    A = GluonField(
                                                        mg=mg,
                                                        k=k3,
                                                        kp=[kp3x, kp3y],
                                                        K=K,
                                                        Kp=Kp,
                                                        a=a,
                                                        pol=pol,
                                                    ).A
                                                    Aslash = (
                                                        0.5 * gamma_plus.dot(A[1])
                                                        - gamma1.dot(A[2])
                                                        - gamma2.dot(A[3])
                                                    )
                                                    for k2 in np.arange(
                                                        -boson_lim, boson_lim + 1, 1
                                                    ):
                                                        for kp2x in range(-Kp, Kp):
                                                            for kp2y in range(-Kp, Kp):
                                                                _psi = QuarkField(
                                                                    m=mq,
                                                                    k=k2,
                                                                    kp=[kp2x, kp2y],
                                                                    K=K,
                                                                    Kp=Kp,
                                                                    c=c2,
                                                                    h=h2,
                                                                ).psi
                                                                term = (
                                                                    g
                                                                    * color_factor
                                                                    * psi_dag.dot(
                                                                        gamma0.dot(
                                                                            Aslash.dot(
                                                                                _psi
                                                                            )
                                                                        )
                                                                    )
                                                                )[0][0]
                                                                # print(
                                                                #     "psi_dag:", psi_dag
                                                                # )
                                                                # print("---")
                                                                # print("Aslash:", Aslash)
                                                                # print("---")
                                                                # print("psi:", _psi)
                                                                print(term)
                                                                ham += term
    ham = ham.normal_order()
    ham.remove_identity()
    ham = (1 / ((2 * L) * (Lp) ** 2)) ** 3 * ham
    return ham


def three_pt_gluon_vertex(K, Kp, Nc, mq, mg, g):
    return


def four_pt_gluon_vertex(K, Kp, Nc, mq, mg, g):
    return


def instantaneous_gluon(K, Kp, Nc, mq, mg, g):
    return


def instantaneous_quark(K, Kp, Nc, mq, mg, g):
    return


def Jplus(q, K, Kp, g):
    Jplus_quark = op.ParticleOperator()
    Jplus_gluon = op.ParticleOperator()

    lim = K - ((K - 0.5) % 1)
    for k in np.arange(-lim, lim + 1, 1):
        for kp1 in range(-Kp, Kp):
            for kp2 in range(-Kp, Kp):
                if q == k1 + k2:
                    pass

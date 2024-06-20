import openparticle as op
import numpy as np


def get_two_body_ham(max_modes):

    # b_p^dag b_q^dag b_r b_s

    H = op.ParticleOperatorSum([])

    for p in range(max_modes):
        for q in range(max_modes):
            for r in range(max_modes):
                for s in range(max_modes):
                    op_string = (
                        "b" + str(p) + "^ b" + str(q) + "^ b" + str(r) + " b" + str(s)
                    )
                    H += op.ParticleOperator(op_string)
    return H

from openparticle.full_dlcq import *
from free_hamiltonian import *
from quark_gluon_vertex import *
from gluon_3pt_vertex import *
from gluon_4pt_vertex import *
from quark_exchange import *
from gluon_exchange import *
from openparticle import ParticleOperator


def qcd_hamiltonian(
    K: float, Kp: float, g: float = 1, mq: float = 1, mg: float = 1e-2, Nc: int = 3
):
    free_quark = dict(free_quark_hamiltonian(K=K, Kp=Kp, mq=mq, Nc=Nc)[0])
    free_gluon = dict(free_gluon_hamiltonian(K=K, Kp=Kp, mg=mg, Nc=Nc)[0])
    quark_gluon_vertex = dict(quark_gluon_vertex_term(K=K, Kp=Kp, g=g, mq=mq, Nc=Nc)[0])
    gluon_3pt_vertex = dict(gluon_3pt_vertex_term(K=K, Kp=Kp, g=g, Nc=Nc)[0])
    gluon_4pt_vertex = dict(gluon_4pt_vertex_term(K=K, Kp=Kp, g=g, Nc=Nc)[0])
    gluon_4pt_inst = dict(gluon_4pt_inst_term(K=K, Kp=Kp, mg=mg, g=g, Nc=Nc)[0])
    quark_4pt_inst = dict(quark_4pt_inst_term(K=K, Kp=Kp, g=g, Nc=Nc, mq=mq, mg=mg)[0])
    quark_gluon_inst = dict(
        quark_gluon_inst_term(K=K, Kp=Kp, g=g, Nc=Nc, mg=mg, mq=mq)[0]
    )
    fermion_exch_inst = dict(fermion_exch_inst_term(K=K, Kp=Kp, g=g, Nc=Nc, mq=mq)[0])

    return (
        ParticleOperator(free_quark)
        + ParticleOperator(free_gluon)
        + ParticleOperator(quark_gluon_vertex)
        + ParticleOperator(gluon_3pt_vertex)
        + ParticleOperator(gluon_4pt_vertex)
        + ParticleOperator(gluon_4pt_inst)
        + ParticleOperator(quark_4pt_inst)
        + ParticleOperator(quark_gluon_inst)
        + ParticleOperator(fermion_exch_inst)
    ).normal_order()

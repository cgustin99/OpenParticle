from openparticle.full_dlcq import *
from free_hamiltonian import *
from quark_gluon_vertex import *
from gluon_3pt_vertex import *
from gluon_4pt_vertex import *
from quark_exchange import *
from gluon_exchange import *
import numba as nb


# @nb.njit(
#     ## output types
#     nb.types.DictType(nb.types.unicode_type, nb.types.complex128)
#     ##input types
#     (nb.float64, nb.float64, nb.float64, nb.float64, nb.int64),
#     ## other options
#     fastmath=True,
# )
def qcd_hamiltonian(
    K: float, Kp: float, g: float = 1, mq: float = 1, mg: float = 1e-2, Nc: int = 3
):
    qcd_ham_list = [
        free_quark_hamiltonian(K=K, Kp=Kp, mq=mq, Nc=Nc),
        free_gluon_hamiltonian(K=K, Kp=Kp, mg=mg, Nc=Nc),
        quark_gluon_vertex_term(K=K, Kp=Kp, g=g, mq=mq, Nc=Nc),
        gluon_3pt_vertex_term(K=K, Kp=Kp, g=g, Nc=Nc),
        gluon_4pt_vertex_term(K=K, Kp=Kp, g=g, Nc=Nc),
        gluon_4pt_inst_term(K=K, Kp=Kp, g=g, Nc=Nc),
        quark_4pt_inst_term(K=K, Kp=Kp, g=g, Nc=Nc, mq=mq, mg=mg),
        quark_gluon_inst_term(K=K, Kp=Kp, g=g, Nc=Nc, mq=mq),
        fermion_exch_inst_term(K=K, Kp=Kp, g=g, Nc=Nc, mq=mg),
    ]

    return qcd_ham_list

import h5py
import json
import os
from datetime import datetime
import time
import argparse

### terminal input
parser = argparse.ArgumentParser()
parser.add_argument("-K", "--res", type=float, help="cut-off resolution")
parser.add_argument("-Kp", "--res_perp", type=float, help="cut-off resolution perp")
parser.add_argument("-mq", "--massq", type=float, help="quark mass")
parser.add_argument("-mg", "--massg", type=float, help="gluon mass")
parser.add_argument("-g", "--coupling", type=float, help="coupling")
args = parser.parse_args()

beginning = time.time()

from quark_exchange import *
from gluon_exchange import *
from quark_gluon_vertex import *
from gluon_3pt_vertex import *
from gluon_4pt_vertex import *
from color_algebra import *
from free_hamiltonian import *
from openparticle.full_dlcq import *

end = time.time()
print("Time to complile Hamiltonian tensor code:", end - beginning)


K = args.res
Kp = args.res_perp
mq = args.massq
mg = args.massg
g = args.coupling

fermion_lim = K - ((K - 0.5) % 1)
fermion_longitudinal_q = np.array(
    [i for i in np.arange(-fermion_lim, fermion_lim + 1.0, 1.0) if i != 0.0],
    dtype=np.float64,
)
boson_lim = int(K)
boson_longitudinal_q = np.array(
    [i for i in np.arange(-boson_lim, boson_lim + 1, 1) if i != 0], dtype=np.float64
)

transverse_q = np.array([i for i in np.arange(-Kp, Kp + 1.0)], dtype=np.float64)
quark_helicities = np.array([1, -1], dtype=np.int8)
quark_colors = np.arange(1, Nc + 1, 1)
gluon_helicities = np.array([1, -1], dtype=np.int8)
gluon_colors = np.arange(1, Nc**2 - 1 + 1, 1)


### SAVE file
cwd = os.getcwd()
data_dir = os.path.join(cwd, "data")
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename_tensor_form = f"qcd_tensor_form_{timestamp}.h5"
filesave_path_tensor_form = os.path.join(data_dir, filename_tensor_form)
filename_hamiltonian = f"qcd_hamiltonian_{timestamp}.txt"
filesave_path_hamiltonian = os.path.join(data_dir, filename_hamiltonian)


start_T1 = time.time()
T1 = free_gluon_hamiltonian_tensor(K=K, Kp=Kp, mg=mg)
end_T1 = time.time()
print("T1 time:", end_T1 - start_T1)
nonzero_idxs1 = T1.nonzero()
nonzero_values1 = T1[nonzero_idxs1]
idxs_t1 = np.block([[nonzero_idxs1[i]] for i in range(len(nonzero_idxs1))]).T
del T1

start_T2 = time.time()
T2 = free_quark_hamiltonian_tensor(K=K, Kp=Kp, mq=mq)
end_T2 = time.time()
print("T2 time:", end_T2 - start_T2)
nonzero_idxs2 = T2.nonzero()
nonzero_values2 = T2[nonzero_idxs2]
idxs_t2 = np.block([[nonzero_idxs2[i]] for i in range(len(nonzero_idxs2))]).T
del T2

start_T3 = time.time()
T3 = quark_gluon_vertex_term_tensor(K=K, Kp=Kp, g=g, mq=mq)
end_T3 = time.time()
print("T3 time:", end_T3 - start_T3)
nonzero_idxs3 = T3.nonzero()
nonzero_values3 = T3[nonzero_idxs3]
idxs_t3 = np.block([[nonzero_idxs3[i]] for i in range(len(nonzero_idxs3))]).T
del T3

start_T4 = time.time()
T4 = gluon_3pt_vertex_term_tensor(K=K, Kp=Kp, g=g)
end_T4 = time.time()
print("T4 time:", end_T4 - start_T4)
nonzero_idxs4 = T4.nonzero()
nonzero_values4 = T4[nonzero_idxs4]
idxs_t4 = np.block([[nonzero_idxs4[i]] for i in range(len(nonzero_idxs4))]).T
del T4

start_T5 = time.time()
T5 = gluon_4pt_vertex_term_tensor(K=K, Kp=Kp, g=g)
end_T5 = time.time()
print("T5 time:", end_T5 - start_T5)
nonzero_idxs5 = T5.nonzero()
nonzero_values5 = T5[nonzero_idxs5]
idxs_t5 = np.block([[nonzero_idxs5[i]] for i in range(len(nonzero_idxs5))]).T
del T5

start_T6 = time.time()
T6 = gluon_legs_term_tensor(K=K, Kp=Kp, g=g, mg=mg)
end_T6 = time.time()
print("T6 time:", end_T6 - start_T6)
nonzero_idxs6 = T6.nonzero()
nonzero_values6 = T6[nonzero_idxs6]
idxs_t6 = np.block([[nonzero_idxs6[i]] for i in range(len(nonzero_idxs6))]).T
del T6

start_T7 = time.time()
T7 = quark_legs_term_tensor(K=K, Kp=Kp, g=g, mg=mg, mq=mq)
end_T7 = time.time()
print("T7 time:", end_T7 - start_T7)
nonzero_idxs7 = T7.nonzero()
nonzero_values7 = T7[nonzero_idxs7]
idxs_t7 = np.block([[nonzero_idxs7[i]] for i in range(len(nonzero_idxs7))]).T
del T7

start_T8 = time.time()
T8 = mixed_legs_term_tensor(K=K, Kp=Kp, g=g, mg=mg, mq=mq)
end_T8 = time.time()
print("T8 time:", end_T8 - start_T8)
nonzero_idxs8 = T8.nonzero()
nonzero_values8 = T8[nonzero_idxs8]
idxs_t8 = np.block([[nonzero_idxs8[i]] for i in range(len(nonzero_idxs8))]).T
del T8

start_T9 = time.time()
T9 = quark_exch_term_tensor(K=K, Kp=Kp, g=g, mq=mg)
end_T9 = time.time()
print("T9 time:", end_T9 - start_T9)
nonzero_idxs9 = T9.nonzero()
nonzero_values9 = T9[nonzero_idxs9]
idxs_t9 = np.block([[nonzero_idxs9[i]] for i in range(len(nonzero_idxs9))]).T
del T9


print(f"saving file at:{filesave_path_tensor_form}")
# Create HDF5 file and add groups
with h5py.File(filesave_path_tensor_form, "w") as f:
    # Create groups
    group1 = f.create_group("free_gluon")
    group1.attrs["description"] = (
        "This contains Hamiltonian elements and q nos for free gluon term"
    )
    dset1 = group1.create_dataset("idxs_t", data=idxs_t1)
    dset2 = group1.create_dataset("nonzero_values", data=nonzero_values1)

    # Create groups
    group2 = f.create_group("free_quark")
    group2.attrs["description"] = (
        "This contains Hamiltonian elements and q nos for free quark term"
    )
    dset3 = group2.create_dataset("idxs_t", data=idxs_t2)
    dset4 = group2.create_dataset("nonzero_values", data=nonzero_values2)

    # Create groups
    group3 = f.create_group("quark_gluon_interaction")
    group3.attrs["description"] = (
        "This contains Hamiltonian elements and q nos for quark gluon vertex"
    )
    dset5 = group3.create_dataset("idxs_t", data=idxs_t3)
    dset6 = group3.create_dataset("nonzero_values", data=nonzero_values3)

    # Create groups
    group4 = f.create_group("ggg_interaction")
    group4.attrs["description"] = (
        "This contains Hamiltonian elements and q nos for 3 gluon vertex"
    )
    dset7 = group4.create_dataset("idxs_t", data=idxs_t4)
    dset8 = group4.create_dataset("nonzero_values", data=nonzero_values4)

    # Create groups
    group5 = f.create_group("gggg_interaction")
    group5.attrs["description"] = (
        "This contains Hamiltonian elements and q nos for 4 gluon vertex"
    )
    dset9 = group5.create_dataset("idxs_t", data=idxs_t5)
    dset10 = group5.create_dataset("nonzero_values", data=nonzero_values5)

    # Create groups
    group6 = f.create_group("gggg_inst_interaction")
    group6.attrs["description"] = (
        "This contains Hamiltonian elements and q nos for 4 gluon inst. vertex"
    )
    dset11 = group6.create_dataset("idxs_t", data=idxs_t6)
    dset12 = group6.create_dataset("nonzero_values", data=nonzero_values6)

    # Create groups
    group7 = f.create_group("qqqq_inst_interaction")
    group7.attrs["description"] = (
        "This contains Hamiltonian elements and q nos for 4 quark inst. vertex"
    )
    dset13 = group7.create_dataset("idxs_t", data=idxs_t7)
    dset14 = group7.create_dataset("nonzero_values", data=nonzero_values7)

    # Create groups
    group8 = f.create_group("mixed_inst_interaction")
    group8.attrs["description"] = (
        "This contains Hamiltonian elements and q nos for mixed inst. vertex"
    )
    dset15 = group8.create_dataset("idxs_t", data=idxs_t8)
    dset16 = group8.create_dataset("nonzero_values", data=nonzero_values8)

    # Create groups
    group9 = f.create_group("quark_inst_interaction")
    group9.attrs["description"] = (
        "This contains Hamiltonian elements and q nos for quark inst. vertex"
    )
    dset17 = group9.create_dataset("idxs_t", data=idxs_t9)
    dset18 = group9.create_dataset("nonzero_values", data=nonzero_values9)

    # Add attributes (metadata)
    group10 = f.create_group("H-params")
    group10.attrs["description"] = "This contains parameters for H setup"

    dset19 = group10.create_dataset("K", data=K)
    dset20 = group10.create_dataset("Kperp", data=Kp)
    dset21 = group10.create_dataset("mg", data=mg)
    dset22 = group10.create_dataset("mq", data=mq)
    dset23 = group10.create_dataset("g", data=g)
    dset24 = group10.create_dataset("runtime", data=end_T9 - start_T1)


# Get QCD Hamiltonian dictionary from nonzero tensor elements in T1->T9
print("Begining to calculate dictionary form...")
start_dict = time.time()
qcd_hamiltonian_dictionary = {}

print("Calculating free quark terms...")
start = time.time()
# Free quark tensor -> qcd hamiltonian dictionary
for i in range(len(nonzero_values2)):
    coeff = nonzero_values2[i]
    qnums = idxs_t2[i]
    qplus = fermion_longitudinal_q[qnums[0]]
    qperp1 = transverse_q[qnums[1]]
    qperp2 = transverse_q[qnums[2]]
    color = quark_colors[qnums[3]]
    helicity = quark_helicities[qnums[4]]
    q = np.array([qplus, qperp1, qperp2], dtype=np.complex128)
    if qplus < 0:
        mode = str(quark_quantum_numbers(k=-q, K=K, Kp=Kp, c=color, h=helicity))
        po_string = "d" + mode + " d" + mode + "^"
    else:
        mode = str(quark_quantum_numbers(k=q, K=K, Kp=Kp, c=color, h=helicity))
        po_string = "b" + mode + "^ b" + mode
    qcd_hamiltonian_dictionary[po_string] = (
        qcd_hamiltonian_dictionary.get(po_string, 0) + coeff
    )
end = time.time()
print("free quark terms runtime: ", end - start)

print("Calculating free gluon terms...")
start = time.time()
# Free gluon tensor -> qcd hamiltonian dictionary
for i in range(len(nonzero_values1)):
    coeff = nonzero_values1[i]
    qnums = idxs_t1[i]
    qplus = fermion_longitudinal_q[qnums[0]]
    qperp1 = transverse_q[qnums[1]]
    qperp2 = transverse_q[qnums[2]]
    color = gluon_colors[qnums[3]]
    helicity = gluon_helicities[qnums[4]]
    q = np.array([qplus, qperp1, qperp2], dtype=np.complex128)
    if qplus < 0:
        mode = str(quark_quantum_numbers(k=-q, K=K, Kp=Kp, c=color, h=helicity))
        po_string = "a" + mode + " a" + mode + "^"
    else:
        mode = str(quark_quantum_numbers(k=q, K=K, Kp=Kp, c=color, h=helicity))
        po_string = "a" + mode + "^ a" + mode
    qcd_hamiltonian_dictionary[po_string] = (
        qcd_hamiltonian_dictionary.get(po_string, 0) + coeff
    )
end = time.time()
print("free gluon terms runtime: ", end - start)

print("Calculating quark-gluon vertex terms...")
start = time.time()
# quark-gluon tensor -> qcd H dict
color_combos_qqg = np.array(
    [
        [0, 0, 2],
        [0, 0, 7],
        [0, 1, 0],
        [0, 1, 1],
        [0, 2, 3],
        [0, 2, 4],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 2],
        [1, 1, 7],
        [1, 2, 5],
        [1, 2, 6],
        [2, 0, 3],
        [2, 0, 4],
        [2, 1, 5],
        [2, 1, 6],
        [2, 2, 7],
    ]
)

spin_arrays_qqg = np.array(
    list(product(quark_helicities, quark_helicities, gluon_polarizations)),
    dtype=np.int8,
)

fixed_qnums_qqg = np.vstack(
    [np.hstack([row1, spin]) for row1 in color_combos_qqg for spin in spin_arrays_qqg]
)

for i in range(len(nonzero_values3)):
    coeff = nonzero_values3[i]
    qnums = idxs_t3[i]

    q1 = np.array(
        [
            fermion_longitudinal_q[qnums[0]],
            transverse_q[qnums[1]],
            transverse_q[qnums[2]],
        ],
        dtype=np.complex128,
    )
    q2 = np.array(
        [
            fermion_longitudinal_q[qnums[3]],
            transverse_q[qnums[4]],
            transverse_q[qnums[5]],
        ],
        dtype=np.complex128,
    )
    q3 = -1 * (q1 + q2)
    fixed = fixed_qnums_qqg[qnums[-1]]
    c1, c2, a = fixed[0], fixed[1], fixed[2]
    h1, h2, p = fixed[3], fixed[4], fixed[5]

    po_string = (
        (
            heaviside(-q1[0])
            * ("b" + str(quark_quantum_numbers(k=-q1, K=K, Kp=Kp, c=c1, h=h1)) + "^ ")
            + heaviside(q1[0])
            * ("d" + str(quark_quantum_numbers(k=q1, K=K, Kp=Kp, c=c1, h=h1)) + " ")
        )
        + (
            heaviside(q2[0])
            * ("b" + str(quark_quantum_numbers(k=q2, K=K, Kp=Kp, c=c2, h=h2)) + " ")
            + heaviside(-q2[0])
            * ("d" + str(quark_quantum_numbers(k=-q2, K=K, Kp=Kp, c=c2, h=h2)) + "^ ")
        )
        + (
            heaviside(q3[0])
            * ("a" + str(gluon_quantum_numbers(k=q3, K=K, Kp=Kp, a=a, pol=p)) + " ")
            + heaviside(-q3[0])
            * ("a" + str(gluon_quantum_numbers(k=-q3, K=K, Kp=Kp, a=a, pol=p)) + "^ ")
        )
    )
    qcd_hamiltonian_dictionary[po_string] = (
        qcd_hamiltonian_dictionary.get(po_string, 0) + coeff
    )
end = time.time()
print("quark-gluon vertex runtime: ", end - start)

print("Computing ggg vertex...")
start = time.time()
# ggg tensor -> qcd H dict
color_combos_ggg = np.array(
    [
        [0, 1, 2],
        [0, 3, 6],
        [0, 4, 5],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 4],
        [2, 5, 6],
        [3, 4, 7],
        [5, 6, 7],
    ]
)

spin_arrays_ggg = np.array(
    list(product(gluon_polarizations, gluon_polarizations, gluon_polarizations)),
    dtype=np.int8,
)

fixed_qnums_ggg = np.vstack(
    [np.hstack([row1, spin]) for row1 in color_combos_ggg for spin in spin_arrays_ggg]
)
for i in range(len(nonzero_values4)):
    coeff = nonzero_values4[i]
    qnums = idxs_t4[i]

    q1 = np.array(
        [
            boson_longitudinal_q[qnums[0]],
            transverse_q[qnums[1]],
            transverse_q[qnums[2]],
        ],
        dtype=np.complex128,
    )
    q2 = np.array(
        [
            boson_longitudinal_q[qnums[3]],
            transverse_q[qnums[4]],
            transverse_q[qnums[5]],
        ],
        dtype=np.complex128,
    )
    q3 = -1 * (q1 + q2)
    fixed = fixed_qnums_ggg[qnums[-1]]
    a, b, c = fixed[0], fixed[1], fixed[2]
    p1, p2, p3 = fixed[3], fixed[4], fixed[5]

    po_string = (
        (
            heaviside(q1[0])
            * ("a" + str(gluon_quantum_numbers(k=q1, K=K, Kp=Kp, a=a, pol=p1)) + " ")
            + heaviside(-q1[0])
            * ("a" + str(gluon_quantum_numbers(k=-q1, K=K, Kp=Kp, a=a, pol=p1)) + "^ ")
        )
        + (
            heaviside(q2[0])
            * ("a" + str(gluon_quantum_numbers(k=q2, K=K, Kp=Kp, a=b, pol=p2)) + " ")
            + heaviside(-q2[0])
            * ("a" + str(gluon_quantum_numbers(k=-q2, K=K, Kp=Kp, a=b, pol=p2)) + "^ ")
        )
        + (
            heaviside(q3[0])
            * ("a" + str(gluon_quantum_numbers(k=q3, K=K, Kp=Kp, a=c, pol=p3)) + " ")
            + heaviside(-q3[0])
            * ("a" + str(gluon_quantum_numbers(k=-q3, K=K, Kp=Kp, a=c, pol=p3)) + "^ ")
        )
    )
    qcd_hamiltonian_dictionary[po_string] = (
        qcd_hamiltonian_dictionary.get(po_string, 0) + coeff
    )

end = time.time()
print("ggg vertex runtime: ", end - start)

print("Calculating gggg vertex...")
start = time.time()
# gggg  tensor -> qcd H dict
color_combos_gggg = np.array(
    [
        [0, 1, 2, 1, 2],
        [0, 1, 2, 3, 6],
        [0, 1, 2, 4, 5],
        [0, 3, 6, 1, 2],
        [0, 3, 6, 3, 6],
        [0, 3, 6, 4, 5],
        [0, 4, 5, 1, 2],
        [0, 4, 5, 3, 6],
        [0, 4, 5, 4, 5],
        [1, 3, 5, 3, 5],
        [1, 3, 5, 4, 6],
        [1, 4, 6, 3, 5],
        [1, 4, 6, 4, 6],
        [2, 3, 4, 3, 4],
        [2, 3, 4, 5, 6],
        [2, 5, 6, 3, 4],
        [2, 5, 6, 5, 6],
        [3, 4, 7, 4, 7],
        [5, 6, 7, 5, 6],
    ]
)

spin_arrays_gggg = np.array(
    list(
        product(
            gluon_polarizations,
            gluon_polarizations,
            gluon_polarizations,
            gluon_polarizations,
        )
    ),
    dtype=np.int8,
)

fixed_qnums_gggg = np.vstack(
    [np.hstack([row1, spin]) for row1 in color_combos_gggg for spin in spin_arrays_gggg]
)
end = time.time()
print("gggg vertex runtime: ", end - start)

print("Calculating gluon exchange terms...")
start = time.time()
# Inst. Gluon Exchange terms
# 1. 4-gluon legs tensor
spin_arrays_gg_g_gg = np.array(
    list(
        product(
            gluon_polarizations,
            gluon_polarizations,
            gluon_polarizations,
            gluon_polarizations,
        )
    ),
    dtype=np.int8,
)

color_combos_gg_g_gg = np.array(
    [
        [0, 1, 2, 1, 2],
        [0, 1, 2, 3, 6],
        [0, 1, 2, 4, 5],
        [0, 3, 6, 1, 2],
        [0, 3, 6, 3, 6],
        [0, 3, 6, 4, 5],
        [0, 4, 5, 1, 2],
        [0, 4, 5, 3, 6],
        [0, 4, 5, 4, 5],
        [1, 3, 5, 3, 5],
        [1, 3, 5, 4, 6],
        [1, 4, 6, 3, 5],
        [1, 4, 6, 4, 6],
        [2, 3, 4, 3, 4],
        [2, 3, 4, 5, 6],
        [2, 5, 6, 3, 4],
        [2, 5, 6, 5, 6],
        [3, 4, 7, 4, 7],
        [5, 6, 7, 5, 6],
    ],
    dtype=np.uint8,
)
fixed_qnums_gg_g_gg = np.vstack(
    [
        np.hstack([row1, spin])
        for row1 in color_combos_gg_g_gg
        for spin in spin_arrays_gg_g_gg
    ]
)
for i in range(len(nonzero_values6)):
    coeff = nonzero_values6[i]
    qnums = idxs_t6[i]

    q1 = np.array(
        [
            boson_longitudinal_q[qnums[0]],
            transverse_q[qnums[1]],
            transverse_q[qnums[2]],
        ],
        dtype=np.complex128,
    )
    q2 = np.array(
        [
            boson_longitudinal_q[qnums[3]],
            transverse_q[qnums[4]],
            transverse_q[qnums[5]],
        ],
        dtype=np.complex128,
    )
    q3 = np.array(
        [
            boson_longitudinal_q[qnums[6]],
            transverse_q[qnums[7]],
            transverse_q[qnums[8]],
        ],
        dtype=np.complex128,
    )
    q4 = -1 * (q1 + q2 + q3)
    fixed = fixed_qnums_gggg[qnums[-1]]
    a, b, c, d, e = fixed[0], fixed[1], fixed[2], fixed[3], fixed[4]
    p1, p2, p3, p4 = fixed[5], fixed[6], fixed[7], fixed[8]

    po_string = (
        (
            heaviside(q1[0])
            * ("a" + str(gluon_quantum_numbers(k=q1, K=K, Kp=Kp, a=b, pol=p1)) + " ")
            + heaviside(-q1[0])
            * ("a" + str(gluon_quantum_numbers(k=-q1, K=K, Kp=Kp, a=b, pol=p1)) + "^ ")
        )
        + (
            heaviside(q2[0])
            * ("a" + str(gluon_quantum_numbers(k=q2, K=K, Kp=Kp, a=c, pol=p2)) + " ")
            + heaviside(-q2[0])
            * ("a" + str(gluon_quantum_numbers(k=-q2, K=K, Kp=Kp, a=c, pol=p2)) + "^ ")
        )
        + (
            heaviside(q3[0])
            * ("a" + str(gluon_quantum_numbers(k=q3, K=K, Kp=Kp, a=d, pol=p3)) + " ")
            + heaviside(-q3[0])
            * ("a" + str(gluon_quantum_numbers(k=-q3, K=K, Kp=Kp, a=d, pol=p3)) + "^ ")
        )
        + (
            heaviside(q4[0])
            * ("a" + str(gluon_quantum_numbers(k=q4, K=K, Kp=Kp, a=e, pol=p4)) + " ")
            + heaviside(-q4[0])
            * ("a" + str(gluon_quantum_numbers(k=-q4, K=K, Kp=Kp, a=e, pol=p4)) + "^ ")
        )
    )
    qcd_hamiltonian_dictionary[po_string] = (
        qcd_hamiltonian_dictionary.get(po_string, 0) + coeff
    )

# 2. quark legs
color_combos_qq_g_qq = np.array(
    [
        [0, 0, 2, 0, 0],
        [0, 0, 2, 1, 1],
        [0, 0, 7, 0, 0],
        [0, 0, 7, 1, 1],
        [0, 0, 7, 2, 2],
        [0, 1, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 2, 3, 0, 2],
        [0, 2, 3, 2, 0],
        [0, 2, 4, 0, 2],
        [0, 2, 4, 2, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 1, 0],
        [1, 0, 1, 0, 1],
        [1, 1, 2, 1, 1],
        [1, 1, 7, 1, 1],
        [1, 2, 5, 1, 2],
        [1, 2, 5, 2, 1],
        [1, 2, 6, 1, 2],
        [1, 2, 6, 2, 1],
        [2, 0, 3, 2, 0],
        [2, 0, 3, 0, 2],
        [2, 0, 4, 2, 0],
        [2, 0, 4, 0, 2],
        [2, 1, 5, 2, 1],
        [2, 1, 5, 1, 2],
        [2, 1, 6, 2, 1],
        [2, 1, 6, 1, 2],
        [2, 2, 7, 2, 2],
    ],
    dtype=np.int8,
)
fixed_qnums_qq_g_qq = np.vstack(
    [
        np.hstack([row1, spin])
        for row1 in color_combos_qq_g_qq
        for spin in spin_arrays_gggg
    ]
)  # Can use same spin_arrays_gggg because spin_arrays_qqqq is the same

for i in range(len(nonzero_values7)):
    coeff = nonzero_values7[i]
    qnums = idxs_t7[i]

    q1 = np.array(
        [
            fermion_longitudinal_q[qnums[0]],
            transverse_q[qnums[1]],
            transverse_q[qnums[2]],
        ],
        dtype=np.complex128,
    )
    q2 = np.array(
        [
            fermion_longitudinal_q[qnums[3]],
            transverse_q[qnums[4]],
            transverse_q[qnums[5]],
        ],
        dtype=np.complex128,
    )
    q3 = np.array(
        [
            fermion_longitudinal_q[qnums[6]],
            transverse_q[qnums[7]],
            transverse_q[qnums[8]],
        ],
        dtype=np.complex128,
    )
    q4 = -1 * (q1 + q2 + q3)
    fixed = fixed_qnums_qq_g_qq[qnums[-1]]
    a, c1, c2, c3, c4 = fixed[2], fixed[0], fixed[1], fixed[3], fixed[4]
    h1, h2, h3, h4 = fixed[5], fixed[6], fixed[7], fixed[8]

    po_string = (
        (
            heaviside(-q1[0])
            * ("b" + str(quark_quantum_numbers(k=-q1, K=K, Kp=Kp, c=c1, h=h1)) + "^ ")
            + heaviside(q1[0])
            * ("d" + str(quark_quantum_numbers(k=q1, K=K, Kp=Kp, c=c1, h=h1)) + " ")
        )
        + (
            heaviside(q2[0])
            * ("b" + str(quark_quantum_numbers(k=q2, K=K, Kp=Kp, c=c2, h=h2)) + " ")
            + heaviside(-q2[0])
            * ("d" + str(quark_quantum_numbers(k=-q2, K=K, Kp=Kp, c=c2, h=h2)) + "^ ")
        )
        + (
            heaviside(-q3[0])
            * ("b" + str(quark_quantum_numbers(k=-q3, K=K, Kp=Kp, c=c3, h=h3)) + "^ ")
            + heaviside(q3[0])
            * ("d" + str(quark_quantum_numbers(k=q3, K=K, Kp=Kp, c=c3, h=h3)) + " ")
        )
        + (
            heaviside(q4[0])
            * ("b" + str(quark_quantum_numbers(k=q4, K=K, Kp=Kp, c=c4, h=h4)) + " ")
            + heaviside(-q4[0])
            * ("d" + str(quark_quantum_numbers(k=-q4, K=K, Kp=Kp, c=c4, h=h4)) + "^ ")
        )
    )
    qcd_hamiltonian_dictionary[po_string] = (
        qcd_hamiltonian_dictionary.get(po_string, 0) + coeff
    )


# 3. Mixed legs
color_combos_qqgg = np.array(
    [
        [0, 0, 2, 3, 4],
        [0, 0, 2, 5, 6],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 3, 6],
        [0, 1, 0, 4, 5],
        [0, 1, 1, 3, 5],
        [0, 1, 1, 4, 6],
        [0, 2, 3, 4, 7],
        [1, 0, 0, 1, 2],
        [1, 0, 0, 3, 6],
        [1, 0, 0, 4, 5],
        [1, 0, 1, 3, 5],
        [1, 0, 1, 4, 6],
        [1, 1, 2, 3, 4],
        [1, 1, 2, 5, 6],
        [1, 2, 5, 6, 7],
        [2, 0, 3, 4, 7],
        [2, 1, 5, 6, 7],
    ]
)
fixed_qnums_qqgg = np.vstack(
    [np.hstack([row1, spin]) for row1 in color_combos_qqgg for spin in spin_arrays_gggg]
)

qcd_hamiltonian_dictionary = {}

for i in range(len(nonzero_values8)):
    coeff = nonzero_values8[i]
    qnums = idxs_t8[i]

    q1 = np.array(
        [
            fermion_longitudinal_q[qnums[0]],
            transverse_q[qnums[1]],
            transverse_q[qnums[2]],
        ],
        dtype=np.complex128,
    )
    q2 = np.array(
        [
            fermion_longitudinal_q[qnums[3]],
            transverse_q[qnums[4]],
            transverse_q[qnums[5]],
        ],
        dtype=np.complex128,
    )
    q3 = np.array(
        [
            boson_longitudinal_q[qnums[6]],
            transverse_q[qnums[7]],
            transverse_q[qnums[8]],
        ],
        dtype=np.complex128,
    )
    q4 = -1 * (q1 + q2 + q3)
    fixed = fixed_qnums_qqgg[qnums[-1]]
    a, c1, c2, b, c = fixed[2], fixed[0], fixed[1], fixed[3], fixed[4]
    h1, h2, p3, p4 = fixed[5], fixed[6], fixed[7], fixed[8]
    po_string = (
        (
            heaviside(-q1[0])
            * ("b" + str(quark_quantum_numbers(k=-q1, K=K, Kp=Kp, c=c1, h=h1)) + "^ ")
            + heaviside(q1[0])
            * ("d" + str(quark_quantum_numbers(k=q1, K=K, Kp=Kp, c=c1, h=h1)) + " ")
        )
        + (
            heaviside(q2[0])
            * ("b" + str(quark_quantum_numbers(k=q2, K=K, Kp=Kp, c=c2, h=h2)) + " ")
            + heaviside(-q2[0])
            * ("d" + str(quark_quantum_numbers(k=-q2, K=K, Kp=Kp, c=c2, h=h2)) + "^ ")
        )
        + (
            heaviside(q3[0])
            * ("a" + str(gluon_quantum_numbers(k=q3, K=K, Kp=Kp, a=b, pol=p3)) + " ")
            + heaviside(-q3[0])
            * ("a" + str(gluon_quantum_numbers(k=-q3, K=K, Kp=Kp, a=b, pol=p3)) + "^ ")
        )
        + (
            heaviside(q4[0])
            * ("a" + str(gluon_quantum_numbers(k=q4, K=K, Kp=Kp, a=c, pol=p4)) + " ")
            + heaviside(-q4[0])
            * ("a" + str(gluon_quantum_numbers(k=-q4, K=K, Kp=Kp, a=c, pol=p4)) + "^ ")
        )
    )
    qcd_hamiltonian_dictionary[po_string] = (
        qcd_hamiltonian_dictionary.get(po_string, 0) + coeff
    )
end = time.time()
print("Inst. gluon exchange terms: ", end - start)

print("Calculating inst. quark exchange...")
start = time.time()

# Inst. quark exchange tensor -> qcd H. Dict
color_combos_qgqg = np.array(
    [
        [0, 0, 2, 2],
        [0, 0, 7, 2],
        [0, 0, 2, 7],
        [0, 0, 7, 7],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 1],
        [0, 2, 3, 3],
        [0, 2, 4, 3],
        [0, 2, 3, 4],
        [0, 2, 4, 4],
        [1, 0, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 1],
        [1, 1, 2, 2],
        [1, 1, 7, 2],
        [1, 1, 2, 7],
        [1, 1, 7, 7],
        [1, 2, 5, 5],
        [1, 2, 6, 5],
        [1, 2, 5, 6],
        [1, 2, 6, 6],
        [2, 0, 3, 3],
        [2, 0, 4, 3],
        [2, 0, 3, 4],
        [2, 0, 4, 4],
        [2, 1, 5, 5],
        [2, 1, 6, 5],
        [2, 1, 5, 6],
        [2, 1, 6, 6],
        [2, 2, 7, 7],
    ]
)

spin_arrays_qgqg = np.array(
    list(
        product(
            quark_helicities, gluon_polarizations, gluon_polarizations, quark_helicities
        )
    ),
    dtype=np.int8,
)

fixed_qnums_qgqg = np.vstack(
    [np.hstack([row1, spin]) for row1 in color_combos_qgqg for spin in spin_arrays_qgqg]
)

qcd_hamiltonian_dictionary = {}

for i in range(len(nonzero_values9)):
    coeff = nonzero_values9[i]
    qnums = idxs_t9[i]

    q1 = np.array(
        [
            fermion_longitudinal_q[qnums[0]],
            transverse_q[qnums[1]],
            transverse_q[qnums[2]],
        ],
        dtype=np.complex128,
    )
    q2 = np.array(
        [
            fermion_longitudinal_q[qnums[3]],
            transverse_q[qnums[4]],
            transverse_q[qnums[5]],
        ],
        dtype=np.complex128,
    )
    q3 = np.array(
        [
            boson_longitudinal_q[qnums[6]],
            transverse_q[qnums[7]],
            transverse_q[qnums[8]],
        ],
        dtype=np.complex128,
    )
    q4 = -1 * (q1 + q2 + q3)
    fixed = fixed_qnums_qgqg[qnums[-1]]
    a, c1, c4, b = fixed[2], fixed[0], fixed[1], fixed[3]
    h1, h4, p2, p3 = fixed[4], fixed[7], fixed[5], fixed[6]
    po_string = (
        (
            heaviside(-q1[0])
            * ("b" + str(quark_quantum_numbers(k=-q1, K=K, Kp=Kp, c=c1, h=h1)) + "^ ")
            + heaviside(q1[0])
            * ("d" + str(quark_quantum_numbers(k=q1, K=K, Kp=Kp, c=c1, h=h1)) + " ")
        )
        + (
            heaviside(q2[0])
            * ("b" + str(quark_quantum_numbers(k=q2, K=K, Kp=Kp, c=c4, h=h4)) + " ")
            + heaviside(-q2[0])
            * ("d" + str(quark_quantum_numbers(k=-q2, K=K, Kp=Kp, c=c4, h=h4)) + "^ ")
        )
        + (
            heaviside(q3[0])
            * ("a" + str(gluon_quantum_numbers(k=q3, K=K, Kp=Kp, a=a, pol=p2)) + " ")
            + heaviside(-q3[0])
            * ("a" + str(gluon_quantum_numbers(k=-q3, K=K, Kp=Kp, a=a, pol=p2)) + "^ ")
        )
        + (
            heaviside(q4[0])
            * ("a" + str(gluon_quantum_numbers(k=q4, K=K, Kp=Kp, a=b, pol=p3)) + " ")
            + heaviside(-q4[0])
            * ("a" + str(gluon_quantum_numbers(k=-q4, K=K, Kp=Kp, a=b, pol=p3)) + "^ ")
        )
    )
    qcd_hamiltonian_dictionary[po_string] = (
        qcd_hamiltonian_dictionary.get(po_string, 0) + coeff
    )
end = time.time()
print("Inst. quark exchange runtime: ", end - start)

end_dict = time.time()

runtime = end_dict - beginning
seconds = int(runtime % 60)
minutes = int((runtime // 60) % 60)
hours = int(runtime // 3600)

qcd_hamiltonian_dictionary_w_parameters = {
    "hamiltonian": qcd_hamiltonian_dictionary,
    "parameters": {"K": K, "Kp": Kp, "mq": mq, "mg": mg, "g": g},
    "runtime": str(hours)
    + "hours, "
    + str(minutes)
    + "minutes, "
    + str(seconds)
    + "seconds.",
}

print(f"Time to calculate full Hamiltonian: {hours} hrs. {minutes} min. {seconds} sec.")
print(f"saving QCD Hamiltonian at:{filesave_path_hamiltonian}")
with open(filesave_path_hamiltonian, "w") as f:
    f.write(str(qcd_hamiltonian_dictionary_w_parameters))

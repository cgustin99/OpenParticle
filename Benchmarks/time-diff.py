#Imports
from openparticle import ParticleOperator
from openfermion import FermionOperator, BosonOperator, normal_ordered
from qiskit_nature.second_q.operators import FermionicOp, BosonicOp
import timeit
import matplotlib.pyplot as plt
import numpy as np

# Define the functions to be tested
def openParticle(op):
    op.normal_order()

def qiskit(op):
    op.normal_order()

def openFermion(op):
    normal_ordered(op)

def random_gen(max_mode, number_of_ops, is_boson):
    allowed_modes = np.arange(max_mode)
    mode_idxs = np.random.choice(allowed_modes, size=number_of_ops)
    creat_idxs = np.random.randint(0,2,size=number_of_ops)

    op = []
    for mode, creat in zip(mode_idxs, creat_idxs):
        if creat == 0:
            op.append(f'a{mode}' if is_boson else f'b{mode}')
        else:
            op.append(f'a{mode}^' if is_boson else f'b{mode}^')
    return " ".join(op) 
        

# Function to run the timeit comparison
def compare_runtimes(max_mode_idx, max_op_len, repeats=100):
    op_len_loop = np.arange(1, max_op_len)

    time_results = np.zeros((4, len(op_len_loop)))
    std_results = np.zeros((4, len(op_len_loop)))

    for idx, op_len in enumerate(op_len_loop):
        print(idx)
        open_f_times_boson = []
        openpart_times_boson = []
        qiskit_times_boson = []

        open_f_times_fermion = []
        openpart_times_fermion = []
        qiskit_times_fermion = []

        for _ in range(repeats):
            boson_case = random_gen(max_mode_idx, op_len, is_boson=True)
            fermion_case = random_gen(max_mode_idx, op_len, is_boson=False)

            boson_op = ParticleOperator(boson_case)
            openpart_times_boson.append(timeit.timeit(lambda: openParticle(boson_op), number=1))

            openfermion_op_string_boson = " ".join(e.replace('a', '') for e in boson_case.split())
            op = BosonOperator(openfermion_op_string_boson)
            open_f_times_boson.append(timeit.timeit(lambda: openFermion(op), number=1))

            qiskit_op_string = " ".join(f"+_{e[1:-1]}" if "^" in e else f"-_{e[1:]}" for e in boson_case.split())
            op_dict = {qiskit_op_string: 1.0}
            op = BosonicOp(op_dict)
            qiskit_times_boson.append(timeit.timeit(lambda: qiskit(op), number=1))


            fermion_op = ParticleOperator(fermion_case)
            openpart_times_fermion.append(timeit.timeit(lambda: openParticle(fermion_op), number=1))

            openfermion_op_string_fermion = " ".join(e.replace('b', '') for e in fermion_case.split())
            op = FermionOperator(openfermion_op_string_fermion)
            open_f_times_fermion.append(timeit.timeit(lambda: openFermion(op), number=1))

            qiskit_op_string = " ".join(f"+_{e[1:-1]}" if "^" in e else f"-_{e[1:]}" for e in fermion_case.split())
            op_dict = {qiskit_op_string: 1.0}
            op = FermionicOp(op_dict)
            qiskit_times_fermion.append(timeit.timeit(lambda: qiskit(op), number=1))

        boson_diff_op_of = [xi - yi for xi, yi in zip(open_f_times_boson, openpart_times_boson)]
        boson_diff_op_qiskit = [xi - yi for xi, yi in zip(qiskit_times_boson, openpart_times_boson)]
        fermion_diff_op_of = [xi - yi for xi, yi in zip(open_f_times_fermion, openpart_times_fermion)]
        fermion_diff_op_qiskit = [xi - yi for xi, yi in zip(qiskit_times_fermion, openpart_times_fermion)]

        ## Y data
        mean_boson_diff_op_of = np.mean(boson_diff_op_of)
        mean_boson_diff_op_qiskit = np.mean(boson_diff_op_qiskit)

        mean_fermion_diff_op_of = np.mean(fermion_diff_op_of)
        mean_fermion_diff_op_qiskit = np.mean(fermion_diff_op_qiskit)

        ## error bar (more like variance)
        std_boson_diff_op_of = np.std(boson_diff_op_of)
        std_boson_diff_op_qiskit = np.std(boson_diff_op_qiskit)

        std_fermion_diff_op_of = np.std(fermion_diff_op_of)
        std_fermion_diff_op_qiskit = np.std(fermion_diff_op_qiskit)

        time_results[0, idx] = mean_boson_diff_op_of
        time_results[1, idx] = mean_boson_diff_op_qiskit
        time_results[2, idx] = mean_fermion_diff_op_of
        time_results[3, idx] = mean_fermion_diff_op_qiskit

        std_results[0, idx] = std_boson_diff_op_of
        std_results[1, idx] = std_boson_diff_op_qiskit
        std_results[2, idx] = std_fermion_diff_op_of
        std_results[3, idx] = std_fermion_diff_op_qiskit

    return op_len_loop, time_results, std_results

# Run the comparison
op_len_loop, time_results, std_results = compare_runtimes(10, 100, repeats=10)

# Plot the results
plt.figure(figsize=(10, 6))
plt.errorbar(op_len_loop, time_results[0,:], yerr=std_results[0,:], label='open fermion vs open particle (boson)', capsize=3, color="orange")
plt.errorbar(op_len_loop, time_results[1,:], yerr=std_results[1,:], label='qiskit vs open particle (boson)', capsize=3, color="dodgerblue")
plt.errorbar(op_len_loop, time_results[2,:], yerr=std_results[2,:], label='open fermion vs open particle (fermion)', capsize=3, color="green")
plt.errorbar(op_len_loop, time_results[3,:], yerr=std_results[3,:], label='qiskit vs open particle (fermion)', capsize=3, color="blueviolet")


plt.xlabel('Input Particle Size')
plt.ylabel('Runtime(s)')
plt.title('Runtime Comparison of Three Functions for both fermions and bosons')
plt.legend()
plt.grid(True)
plt.show()

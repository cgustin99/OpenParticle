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

    time_results = np.zeros((6, len(op_len_loop)))
    std_results = np.zeros((6, len(op_len_loop)))

    for idx, op_len in enumerate(op_len_loop):
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


        ## Y data
        mean_times_of_boson = np.mean(open_f_times_boson)
        mean_times_op_boson = np.mean(openpart_times_boson)
        mean_times_qiskit_boson = np.mean(qiskit_times_boson)

        mean_times_of_fermion = np.mean(open_f_times_fermion)
        mean_times_op_fermion = np.mean(openpart_times_fermion)
        mean_times_qiskit_fermion = np.mean(qiskit_times_fermion)

        ## error bar (more like variance)
        std_times_of_boson = np.std(open_f_times_boson)
        std_times_op_boson = np.std(openpart_times_boson)
        std_times_qiskit_boson = np.std(qiskit_times_boson)

        std_times_of_fermion = np.std(open_f_times_fermion)
        std_times_op_fermion = np.std(openpart_times_fermion)
        std_times_qiskit_fermion = np.std(qiskit_times_fermion)

        time_results[0, idx] = mean_times_of_boson
        time_results[1, idx] = mean_times_op_boson
        time_results[2, idx] = mean_times_qiskit_boson
        time_results[3, idx] = mean_times_of_fermion
        time_results[4, idx] = mean_times_op_fermion
        time_results[5, idx] = mean_times_qiskit_fermion

        std_results[0, idx] = std_times_of_boson
        std_results[1, idx] = std_times_op_boson
        std_results[2, idx] = std_times_qiskit_boson
        std_results[3, idx] = std_times_of_fermion
        std_results[4, idx] = std_times_op_fermion
        std_results[5, idx] = std_times_qiskit_fermion

    return op_len_loop, time_results, std_results

# Run the comparison
op_len_loop, time_results, std_results = compare_runtimes(10, 100, repeats=10)

# Plot the results
plt.figure(figsize=(10, 6))
plt.errorbar(op_len_loop, time_results[0,:], yerr=std_results[0,:], label='open fermion (boson)', capsize=3, color="orange")
plt.errorbar(op_len_loop, time_results[1,:], yerr=std_results[1,:], label='open particle (boson)', capsize=3, color="dodgerblue")
plt.errorbar(op_len_loop, time_results[2,:], yerr=std_results[2,:], label='qiskit (boson)', capsize=3, color="green")
plt.errorbar(op_len_loop, time_results[3,:], yerr=std_results[3,:], label='open fermion (fermion)', capsize=3, color="goldenrod")
plt.errorbar(op_len_loop, time_results[4,:], yerr=std_results[4,:], label='open particle (fermion)', capsize=3, color="deepskyblue")
plt.errorbar(op_len_loop, time_results[5,:], yerr=std_results[5,:], label='qiskit (fermion)', capsize=3, color="mediumseagreen")


plt.xlabel('Input Particle Size')
plt.ylabel('Runtime(s)')
plt.title('Runtime Comparison of Three Functions for both fermions and bosons')
plt.legend()
plt.grid(True)
plt.show()

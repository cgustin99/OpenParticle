#Imports
from openparticle import ParticleOperator
from openfermion import FermionOperator, normal_ordered
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

def random_gen(max_mode, number_of_ops):
    # max_mode = 10000
    # number_of_ops = 100
    allowed_modes = np.arange(max_mode)
    mode_idxs = np.random.choice(allowed_modes, size=number_of_ops)
    creat_idxs = np.random.randint(0,2,size=number_of_ops)

    op = []
    for mode, creat in zip(mode_idxs, creat_idxs):
        if creat == 0:
            op.append(f'b{mode}')
        else:
            op.append(f'b{mode}^')
    return " ".join(op) 
        

# Function to run the timeit comparison
def compare_runtimes(max_mode_idx, max_op_len, repeats=100):
    test_cases = []
    results = []
    # for case in test_cases:

    op_len_loop = np.arange(1, max_op_len)

    time_results = np.zeros((3, len(op_len_loop)))
    std_results = np.zeros((3, len(op_len_loop)))
    # for _ in range(loop):
        # Generate a random test case
        
    for idx, op_len in enumerate(op_len_loop):
        # op = ParticleOperator().random(["fermion"], 1, 400, 400, False, False)
        # case = list(op.op_dict.keys())[0]
        # test_cases.append(case)

        open_f_times = []
        openpart_times = []
        qiskit_times = []

        for _ in range(repeats):
            case = random_gen(max_mode_idx, op_len)
            op = ParticleOperator(case)
            openpart_times.append(timeit.timeit(lambda: openParticle(op), number=1))

            fermion_op_string = " ".join(e.replace('b', '') for e in case.split())
            op = FermionOperator(fermion_op_string)
            open_f_times.append(timeit.timeit(lambda: openFermion(op), number=1))

            qiskit_op_string = " ".join(f"+_{e[1:-1]}" if "^" in e else f"-_{e[1:]}" for e in case.split())
            op_dict = {qiskit_op_string: 1.0}
            op = FermionicOp(op_dict)
            qiskit_times.append(timeit.timeit(lambda: qiskit(op), number=1))


        ## Y data
        mean_times_of = np.mean(open_f_times)
        mean_times_op = np.mean(openpart_times)
        mean_times_qiskit = np.mean(qiskit_times)

        ## error bar (more like variance)
        std_times_of = np.std(open_f_times)
        std_times_op = np.std(openpart_times)
        std_times_qiskit = np.std(qiskit_times)

        time_results[0, idx] = mean_times_of
        time_results[1, idx] = mean_times_op
        time_results[2, idx] = mean_times_qiskit

        std_results[0, idx] = std_times_of
        std_results[1, idx] = std_times_op
        std_results[2, idx] = std_times_qiskit

    return op_len_loop, time_results, std_results

# Run the comparison
op_len_loop, time_results, std_results = compare_runtimes(100, 600, repeats=10)

# Plot the results
plt.figure(figsize=(10, 6))
# plt.plot(input_sizes, times_func1, label='open particle')
# plt.plot(input_sizes, times_func2, label='qiskit')
# plt.plot(input_sizes, times_func3, label='open fermion')

# plt.scatter(input_sizes, times_func1)
# plt.scatter(input_sizes, times_func2)
# plt.scatter(input_sizes, times_func3)


plt.errorbar(op_len_loop, time_results[1,:], yerr=std_results[1,:], label='open particle', capsize=3)
plt.errorbar(op_len_loop, time_results[2,:], yerr=std_results[2,:], label='qiskit', capsize=3)
plt.errorbar(op_len_loop, time_results[0,:], yerr=std_results[0,:], label='open fermion', capsize=3)



plt.xlabel('Input Particle Size')
plt.ylabel('Runtime(s)')
plt.title('Runtime Comparison of Three Functions')
plt.legend()
plt.grid(True)
plt.show()



# # Extract data for plotting
# input_sizes = [result[0] for result in results]
# times_func1 = [result[1] for result in results]
# times_func2 = [result[2] for result in results]
# times_func3 = [result[3] for result in results]

# # Plot the results
# plt.figure(figsize=(10, 6))
# # plt.plot(input_sizes, times_func1, label='open particle')
# # plt.plot(input_sizes, times_func2, label='qiskit')
# # plt.plot(input_sizes, times_func3, label='open fermion')

# # plt.scatter(input_sizes, times_func1)
# # plt.scatter(input_sizes, times_func2)
# # plt.scatter(input_sizes, times_func3)


# plt.scatter(input_sizes, times_func1, label='open particle')
# plt.scatter(input_sizes, times_func2, label='qiskit')
# plt.scatter(input_sizes, times_func3, label='open fermion')

# plt.xlabel('Input Particle Size')
# plt.ylabel('Runtime(s)')
# plt.title('Runtime Comparison of Three Functions')
# plt.legend()
# plt.grid(True)
# plt.show()

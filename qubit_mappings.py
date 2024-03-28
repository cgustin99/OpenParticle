import numpy as np
from symmer import PauliwordOp as Pauli
from ParticleOperator import *
from FockState import *
from typing import Union
from symmer.utils import tensor_list


def jordan_wigner(op: Union[ParticleOperator, FermionOperator, 
                            AntifermionOperator], total_modes):

    assert op.modes[0] <= total_modes

    if isinstance(op, (FermionOperator, AntifermionOperator)):
        qubit_op_list = ['Z' * op.modes[0] + 'X', 'Z' * op.modes[0] + 'Y']
        
        if op.ca_string == 'c':
            coeffs = [1/2, -1j/2]
        elif op.ca_string == 'a': 
            coeffs = [1/2, 1j/2]

        op = Pauli.from_list(qubit_op_list, coeffs)
        
        return op.tensor(Pauli.from_list(['I' * (total_modes - op.n_qubits)]))
    
    else: raise Exception("The Jordan Wigner mapping only works for fermions and antifermions")


def somma(op: Union[BosonOperator, ParticleOperator], max_bose_mode_occ: int, total_modes: int):

    # assert op.modes[0] <= total_modes

    if isinstance(op, BosonOperator):
        p = op.modes[0]

        Mb = max_bose_mode_occ

        pauli_list = []
        coeffs_list = []
        nq_max = (p + 1) * (Mb + 1) 

        for j in range(0, Mb):
            q = (Mb + 1) * p + j
            qubit_diff = nq_max - q - 2
            pauli_list += ['I' * q + 'XX' + "I" * qubit_diff,
                            'I' * q + "XY" + "I" * qubit_diff,
                            'I' * q + "YX" + "I" * qubit_diff,
                            'I' * q + "YY" + "I" * qubit_diff]
            if op.ca_string == 'c':
                coeffs_list += list(np.sqrt(j + 1) / 4 * np.array([1, 1j, -1j, 1]))
            elif op.ca_string == 'a':
                coeffs_list += list(np.sqrt(j + 1) / 4 * np.array([1, -1j, 1j, 1]))

        op = Pauli.from_list(pauli_list, coeffs_list)
        qubit_diff = (total_modes + 1) * (Mb + 1) - op.n_qubits
        return op.tensor(Pauli.from_list(['I' * qubit_diff], [1]))

    else: raise NotImplemented


def qubit_op_mapping(op: Union[ParticleOperator, FermionOperator, AntifermionOperator, BosonOperator], 
                  total_modes_list: List,
                  max_bose_mode_occ = None):
    
    if isinstance(op, FermionOperator): return jordan_wigner(op)
    elif isinstance(op, AntifermionOperator): return jordan_wigner(op)
    elif isinstance(op, BosonOperator): 
        if max_bose_mode_occ == None: raise Exception("Must provide maximum bosonic mode occupancy")
        else: return somma(op, max_bose_mode_occ)

    elif isinstance(op, ParticleOperator):

        ops = []
        for index, particle in enumerate(op.particle_str):
            if particle == 'f':
                qubit_op = jordan_wigner(FermionOperator(op.modes[index], 
                                                    op.ca_string[index]), total_modes_list[0])
            elif particle == 'a':
                qubit_op = jordan_wigner(AntifermionOperator(op.modes[index], 
                                                    op.ca_string[index]), total_modes_list[1])
            elif particle == 'b':
                qubit_op = somma(BosonOperator(op.modes[index], 
                                                    op.ca_string[index]), max_bose_mode_occ, total_modes_list[2])
            ops.append(qubit_op)

        
    return tensor_list(ops)




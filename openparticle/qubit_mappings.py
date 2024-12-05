import numpy as np
from symmer import PauliwordOp, QuantumState
from openparticle import (
    ParticleOperator,
    FermionOperator,
    AntifermionOperator,
    BosonOperator,
)
from typing import Union
from symmer.utils import tensor_list


def SB(operator: BosonOperator, max_occ: int):

    assert np.log2(max_occ + 1) % 1 == 0, "Max occupancy must be given as 2^N - 1"

    fock_to_qubit = {
        (1, 0): PauliwordOp.from_list(["X", "Y"], [1 / 2, -1j / 2]),
        (0, 1): PauliwordOp.from_list(["X", "Y"], [1 / 2, 1j / 2]),
        (0, 0): PauliwordOp.from_list(["I", "Z"], [1 / 2, 1 / 2]),
        (1, 1): PauliwordOp.from_list(["I", "Z"], [1 / 2, -1 / 2]),
    }
    n_qubits = int(np.log2(max_occ + 1))

    output = PauliwordOp.empty(n_qubits=n_qubits)
    for s in range(0, max_occ):
        if operator.all_creation:

            ket_str = f"{s + 1:0{n_qubits}b}"
            bra_str = f"{s:0{n_qubits}b}"
        else:
            ket_str = f"{s:0{n_qubits}b}"
            bra_str = f"{s + 1:0{n_qubits}b}"

        list_to_tensor = []
        for bit in range(len(ket_str)):
            list_to_tensor.append(fock_to_qubit[(int(ket_str[bit]), int(bra_str[bit]))])

        output += tensor_list(list_to_tensor) * np.sqrt(s + 1)

    return output


def jordan_wigner(operator: Union[FermionOperator, AntifermionOperator]):

    if operator.all_creation:
        return PauliwordOp.from_list(["X", "Y"], [1 / 2, -1j / 2])
    else:
        return PauliwordOp.from_list(["X", "Y"], [1 / 2, 1j / 2])


def op_qubit_map(operator, max_bose_occ: int = None):

    if operator.has_fermions:
        n_fermionic_qubits = operator.max_fermionic_mode + 1
    else:
        n_fermionic_qubits = 0

    if operator.has_antifermions:
        n_antifermionic_qubits = operator.max_antifermionic_mode + 1
    else:
        n_antifermionic_qubits = 0

    if operator.has_bosons:
        n_boson_qubits_one_mode = int(np.log2(max_bose_occ + 1))
        n_bosonic_qubits = (operator.max_bosonic_mode + 1) * n_boson_qubits_one_mode
    else:
        n_boson_qubits_one_mode = 1
        n_bosonic_qubits = 0

    empty = PauliwordOp.empty(
        n_qubits=n_fermionic_qubits + n_antifermionic_qubits + n_bosonic_qubits
    )

    for op in operator.to_list():
        fermion_op_list = [PauliwordOp.from_list(["I"], [1])] * n_fermionic_qubits
        antifermion_op_list = [
            PauliwordOp.from_list(["I"], [1])
        ] * n_antifermionic_qubits
        boson_op_list = [
            PauliwordOp.from_list(["I" * n_boson_qubits_one_mode], [1])
        ] * int(n_bosonic_qubits / n_boson_qubits_one_mode)
        for spilt_op in op.split():
            if isinstance(spilt_op, FermionOperator):
                fermion_op_list[
                    n_fermionic_qubits - 1 - spilt_op.mode
                ] *= jordan_wigner(spilt_op)
            elif isinstance(spilt_op, AntifermionOperator):
                antifermion_op_list[
                    n_antifermionic_qubits - 1 - spilt_op.mode
                ] *= jordan_wigner(spilt_op)
            elif isinstance(spilt_op, BosonOperator):
                boson_op_list[spilt_op.mode] *= SB(spilt_op, max_bose_occ)

        empty += tensor_list(fermion_op_list + antifermion_op_list + boson_op_list)

    return empty


def fermion_fock_to_qubit(f_occ, max_mode: int = None):
    if f_occ != []:
        fock_list = f_occ
        if max_mode is not None:
            qubit_state = [0] * (max_mode + 1)
        else:
            qubit_state = [0] * (max(f_occ) + 1)

        for index in fock_list:
            qubit_state[index] = 1
        return qubit_state[::-1]

    else:
        if max_mode is not None:
            return [0] * (max_mode + 1)
        else:
            return []


def boson_fock_to_qubit(b_occ, max_occ, max_mode: int = None):
    if b_occ != []:
        if max_mode is not None:
            _max = max_mode
        else:
            _max = max(b_occ, key=lambda x: x[0])[0]
        n_qubits_one_mode = int(np.log2(max_occ + 1))
        qubit_occ = ["0" * n_qubits_one_mode] * (_max + 1)
        for occ in b_occ:
            qubit_occ[_max - occ[0]] = f"{occ[1]:0{n_qubits_one_mode}b}"

        return [int(char) for segment in qubit_occ for char in segment]

    else:
        if max_mode is not None:
            return [0] * (max_mode + 1) * int(np.log2(max_occ + 1))
        else:
            return []


def fock_qubit_state_mapping(
    state,
    max_occ: int,
    max_fermion_mode: int = None,
    max_antifermion_mode: int = None,
    max_boson_mode: int = None,
):
    qubit_state = (
        fermion_fock_to_qubit(state.f_occ, max_mode=max_fermion_mode)
        + fermion_fock_to_qubit(state.af_occ, max_mode=max_antifermion_mode)
        + boson_fock_to_qubit(state.b_occ, max_occ=max_occ, max_mode=max_boson_mode)
    )
    return QuantumState(qubit_state)

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


def _jordan_wigner(operator: Union[FermionOperator, AntifermionOperator]):

    if operator.all_creation:
        return PauliwordOp.from_list(["X", "Y"], [1 / 2, -1j / 2])
    else:
        return PauliwordOp.from_list(["X", "Y"], [1 / 2, 1j / 2])


def jordan_wigner(operator: ParticleOperator):
    assert not operator.has_bosons, "Must be a fermionic operator only"

    n_modes = operator.max_mode + 1
    jw = PauliwordOp.from_dictionary({"I" * (n_modes): 1.0})

    for term in operator.to_list():
        _jw = PauliwordOp.from_dictionary({"I" * (n_modes): 1.0})
        for op in term.split():
            mode = op.mode
            n_higher_modes = n_modes - (mode + 1)
            n_lower_modes = mode
            current_jw = tensor_list(
                [PauliwordOp.from_list(["I"], [1])] * n_higher_modes
                + [_jordan_wigner(op)]
                + [PauliwordOp.from_list(["Z"], [1])] * n_lower_modes
            )
            _jw *= current_jw

    jw *= _jw
    return jw


def op_qubit_map(operator, max_bose_occ: int = None):

    for term in operator.to_list():
        mapped_fermions = []
        mapped_antifermions = []
        mapped_bosons = []
        for op in term.partition():

            if op.has_fermions:
                mapped_fermions.append(jordan_wigner(op))
            elif op.has_antifermions:
                mapped_antifermions.append(jordan_wigner(op))
            elif op.has_bosons:
                mapped_bosons.append(SB(op, max_occ=max_bose_occ))

    return tensor_list(mapped_fermions + mapped_antifermions + mapped_bosons)


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
    max_occ: int = None,
    max_fermion_mode: int = None,
    max_antifermion_mode: int = None,
    max_boson_mode: int = None,
):
    qubit_f_occ = []
    qubit_af_occ = []
    qubit_b_occ = []

    if state.has_fermions:
        qubit_f_occ = fermion_fock_to_qubit(state.f_occ, max_mode=max_fermion_mode)
    if state.has_antifermions:
        qubit_af_occ = fermion_fock_to_qubit(
            state.af_occ, max_mode=max_antifermion_mode
        )
    if state.has_bosons:
        qubit_b_occ = boson_fock_to_qubit(
            state.b_occ, max_occ=max_occ, max_mode=max_boson_mode
        )

    qubit_state = qubit_f_occ + qubit_af_occ + qubit_b_occ
    return QuantumState(qubit_state)

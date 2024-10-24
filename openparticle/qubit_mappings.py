import numpy as np
from symmer import PauliwordOp as Pauli
from symmer import QuantumState
from openparticle import (
    Fock,
    ParticleOperator,
    FermionOperator,
    AntifermionOperator,
    BosonOperator,
)
from typing import Union, List
from symmer.utils import tensor_list
import symmer
from IPython.display import display, Latex


def jordan_wigner(
    op: Union[FermionOperator, AntifermionOperator, ParticleOperator],
    total_modes: int = None,
    display_latex: bool = False,
):

    if isinstance(op, ParticleOperator):
        particle = next(iter(op.op_dict))[0]
        if next(iter(op.op_dict))[-1] == "^":
            create = True
            mode = next(iter(op.op_dict))[1:-1]
        else:
            create = False
            mode = next(iter(op.op_dict))[1:]

        op_string = mode + "^" * create

        if particle == "b":
            op = FermionOperator(op_string)
        elif particle == "d":
            op = AntifermionOperator(op_string)

    qubit_op_string = ""

    qubit_op_list = ["X" + "Z" * op.mode, "Y" + "Z" * op.mode]

    if op.creation:
        coeffs = [1 / 2, -1j / 2]
        qubit_op_string = (
            "$" + str(0.5) + qubit_op_list[0] + " - 0.5" + "i" + qubit_op_list[1] + "$"
        )
    elif not op.creation:
        coeffs = [1 / 2, 1j / 2]
        qubit_op_string = (
            "$" + str(0.5) + qubit_op_list[0] + " + 0.5" + "i" + qubit_op_list[1] + "$"
        )

    if display_latex:
        display(Latex(qubit_op_string))

    op = Pauli.from_list(qubit_op_list, coeffs)
    if total_modes is not None:
        return Pauli.from_list(["I" * (total_modes - op.n_qubits)]).tensor(op)
    else:
        return op


def unary(
    op: BosonOperator,
    max_bose_mode_occ: int,
    total_modes: int = None,
    display_latex: bool = False,
):

    # assert op.modes[0] <= total_modes

    p = op.mode

    Mb = max_bose_mode_occ

    pauli_list = []
    coeffs_list = []
    nq_max = (p + 1) * (Mb + 1)

    op_str = ""

    for j in range(0, Mb):
        q = (Mb + 1) * p + j
        qubit_diff = nq_max - q - 2
        pauli_list += [
            "I" * qubit_diff + "XX" + "I" * q,
            "I" * qubit_diff + "XY" + "I" * q,
            "I" * qubit_diff + "YX" + "I" * q,
            "I" * qubit_diff + "YY" + "I" * q,
        ]
        pauli_list_static = [
            "I" * qubit_diff + "XX" + "I" * q,
            "I" * qubit_diff + "XY" + "I" * q,
            "I" * qubit_diff + "YX" + "I" * q,
            "I" * qubit_diff + "YY" + "I" * q,
        ]
        if op.creation:
            coeffs_list += list(np.sqrt(j + 1) / 4 * np.array([1, 1j, -1j, 1]))
            op_str += (
                str(round(np.sqrt(j + 1) / 4, 3))
                + pauli_list_static[0]
                + " + "
                + str(round(np.sqrt(j + 1) / 4, 3))
                + "i"
                + pauli_list_static[1]
                + " - "
                + str(round(np.sqrt(j + 1) / 4, 3))
                + "i"
                + pauli_list_static[2]
                + " + "
                + str(round(np.sqrt(j + 1) / 4, 3))
                + pauli_list_static[3]
                + " +"
            )
        elif not op.creation:
            coeffs_list += list(np.sqrt(j + 1) / 4 * np.array([1, -1j, 1j, 1]))
            op_str += (
                str(round(np.sqrt(j + 1) / 4, 3))
                + pauli_list_static[0]
                + " - "
                + str(round(np.sqrt(j + 1) / 4, 3))
                + "i"
                + pauli_list_static[1]
                + " + "
                + str(round(np.sqrt(j + 1) / 4, 3))
                + "i"
                + pauli_list_static[2]
                + " + "
                + str(round(np.sqrt(j + 1) / 4, 3))
                + pauli_list_static[3]
                + " +"
            )

    op = Pauli.from_list(pauli_list, coeffs_list)
    if display_latex:
        op_str = "$" + op_str[:-1] + "$"
        display(Latex(op_str))

    if total_modes is not None:
        qubit_diff = (total_modes + 1) * (Mb + 1) - op.n_qubits
        return Pauli.from_list(["I" * qubit_diff], [1]).tensor(op)
    else:
        return op


def qubit_op_mapping(
    op: Union[ParticleOperator, FermionOperator, AntifermionOperator, BosonOperator],
    max_bose_mode_occ: int = None,
    total_modes: int = None,
):

    if isinstance(op, FermionOperator):
        return jordan_wigner(op)
    elif isinstance(op, AntifermionOperator):
        return jordan_wigner(op)
    elif isinstance(op, BosonOperator):
        if max_bose_mode_occ == None:
            raise Exception("Must provide maximum bosonic mode occupancy")
        else:
            return unary(op, max_bose_mode_occ)

    elif isinstance(op, ParticleOperator):

        if op.has_fermions:
            fermion_qubit = Pauli.from_list(["I" * (op.max_fermionic_mode + 1)], [1])
        else:
            fermion_qubit = Pauli.empty()
        if op.has_antifermions:
            antifermion_qubit = Pauli.from_list(
                ["I" * (op.max_antifermionic_mode + 1)], [1]
            )
        if op.has_bosons:
            boson_qubit = Pauli.from_list(
                ["I" * ((op.max_bosonic_mode + 1) * (max_bose_mode_occ + 1))], [1]
            )

        ops_to_tensor = []
        for element in op.split():
            if isinstance(element, FermionOperator):
                fermion_qubit *= jordan_wigner(
                    FermionOperator(str(element.mode) + "^" * element.creation),
                    op.max_fermionic_mode + 1,
                )
            elif isinstance(element, AntifermionOperator):
                antifermion_qubit *= jordan_wigner(
                    AntifermionOperator(str(element.mode) + "^" * element.creation),
                    op.max_antifermionic_mode + 1,
                )
            elif isinstance(element, BosonOperator):
                boson_qubit *= unary(
                    BosonOperator(str(element.mode) + "^" * element.creation),
                    max_bose_mode_occ,
                    op.max_bosonic_mode,
                )

        if "fermion_qubit" in vars():
            ops_to_tensor.append(fermion_qubit)
        if "antifermion_qubit" in vars():
            ops_to_tensor.append(antifermion_qubit)
        if "boson_qubit" in vars():
            ops_to_tensor.append(boson_qubit)

        return tensor_list(ops_to_tensor)


def qubit_state_mapping(state, max_bose_mode_occ):

    q_fermi = map_fermions_to_qubits(state)
    q_antifermi = map_antifermions_to_qubits(state)

    q_bos = map_bose_occ(
        list(state.state_dict.keys())[0][2], max_bose_mode_occ  # b_occ
    )

    return symmer.QuantumState([q_fermi + q_antifermi + q_bos])


def map_bose_occ(occupancy_list, M):
    modes = [i[0] for i in occupancy_list]
    q_bos = [0] * (max(modes) + 1) * (M + 1)

    for occ in occupancy_list:
        p = occ[0]
        q = occ[1]
        index = (M + 1) * p + q
        q_bos[index] = 1

    return q_bos[::-1]


def map_fermions_to_qubits(state):
    f_occ = list(state.state_dict.keys())[0][0]
    if f_occ != []:
        fock_list = f_occ
        qubit_state = [0] * (f_occ[-1] + 1)

        for index in fock_list:
            qubit_state[index] = 1
        return qubit_state[::-1]

    else:
        return []


def map_antifermions_to_qubits(state):
    af_occ = list(state.state_dict.keys())[0][1]
    if af_occ != []:
        fock_list = af_occ
        qubit_state = [0] * (af_occ[-1] + 1)

        for index in fock_list:
            qubit_state[index] = 1
        return qubit_state[::-1]

    else:
        return []

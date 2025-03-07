import numpy as np
from typing import List, Union, Optional, Dict, Tuple, Literal
from IPython.display import display, Latex
from collections import defaultdict, Counter
import re
from copy import deepcopy
from itertools import product
import math
import scipy

from symmer import PauliwordOp, QuantumState
from symmer.utils import tensor_list


class ParticleOperator:

    def __init__(
        self, operator: Union[Dict[str, complex], str] = dict(), coeff: complex = 1.0
    ):

        if isinstance(operator, str):
            key, val = ParticleOperator.op_string_to_key(operator, coeff)
            self.op_dict = {key: val}
        elif isinstance(operator, dict):
            if operator == {}:
                self.op_dict = {}
            else:
                # iterate over terms in dictionary
                self.op_dict = {}
                for i, j in operator.items():
                    if isinstance(i, tuple):
                        self.op_dict[i] = j
                    elif isinstance(i, str):
                        key, val = ParticleOperator.op_string_to_key(i, j)
                        self.op_dict[key] = val
        else:
            raise ValueError("input must be dictionary or op string")

        self.items = list(self.op_dict.items())
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.items):
            key, value = self.items[self.index]
            self.index += 1
            return ParticleOperator({key: value})
        else:
            raise StopIteration

    @classmethod
    def from_openfermion(cls, of_operator):
        """
        Initializes a ParticleOperator from a openfermion FermionOperator instance
        Args:
            of_operator: openfermion.FermionOperator object
        Returns:
            ParticleOperator: A new ParticleOperator object matching the input
        """
        from openfermion import FermionOperator as fo

        assert isinstance(of_operator, fo), "Must supply a FermionOperator"

        op_dict = {
            tuple((0,) + inner_tuple for inner_tuple in key): value
            for key, value in of_operator.terms.items()
        }
        return ParticleOperator(op_dict)

    @staticmethod
    def op_string_to_key(op_str, coeff: complex = 1.0):
        key = []

        for op in op_str.split():
            if op[-1] == "^":
                create_or_annihilate = 1
                mode = op[1:-1]
            else:
                create_or_annihilate = 0
                mode = op[1:]
            if op[0] == "b":
                particle_type = 0
            elif op[0] == "d":
                particle_type = 1
            elif op[0] == "a":
                particle_type = 2
            key.append((particle_type, int(mode), create_or_annihilate))

        return tuple(key), coeff

    @staticmethod
    def key_to_op_string(key):
        op_str = ""
        for op in key:
            if op[0] == 0:
                op_str += "b"
            elif op[0] == 1:
                op_str += "d"
            elif op[0] == 2:
                op_str += "a"

            if op[2] == 1:
                op_str += str(op[1]) + "^ "
            elif op[2] == 0:
                op_str += str(op[1]) + " "
        return op_str[:-1]

    def __add__(self, other: "ParticleOperator") -> "ParticleOperator":
        new_dict = defaultdict(float, self.op_dict)
        for key, value in other.op_dict.items():
            new_dict[key] += value
        return ParticleOperator(new_dict)._cleanup()

    def __str__(self) -> str:
        if len(self.op_dict) == 0:
            return "0"
        else:
            output_str = ""
            for op, coeff in self.op_dict.items():
                output_str += f"{coeff} * {ParticleOperator.key_to_op_string(op)}"
                output_str += "\n"
            return output_str

    def _cleanup(self, zero_threshold=1e-128) -> "ParticleOperator":
        """
        remove terms below threshold
        """
        if len(self.op_dict) == 0:
            return self
        else:
            keys, coeffs = zip(*self.op_dict.items())
            mask = np.where(abs(np.array(coeffs)) > zero_threshold)[0]

            new_keys = np.array(keys, dtype=object)
            if len(new_keys.shape) == 1:
                new_op_dict = dict(
                    zip(
                        np.take(np.array(keys, dtype=object), mask),
                        np.take(coeffs, mask),
                    )
                )
            else:
                ## odd edge case where only have single terms
                new_op_dict = {}
                for idx in mask:
                    new_op_dict[keys[idx]] = coeffs[idx]
            return ParticleOperator(new_op_dict)

    def __repr__(self) -> str:
        return self.__str__()

    def latex_string(self) -> Latex:
        pattern = r"(\d+)"
        replacement = r"_{\1}"
        latex_str = ""
        for s, coeff in self.op_dict.items():
            new_s = s.replace("^", "^†")
            latex_str += f"{coeff} {re.sub(pattern, replacement, new_s)} +"

        return Latex("$" + latex_str[:-1] + "$")

    def display(self):
        display(self.latex_string())

    def split(self) -> List:
        split_list = []
        if len(self.op_dict) == 1:
            for oper in ParticleOperator.key_to_op_string(
                list(self.op_dict.keys())[0]
            ).split(" "):
                if len(oper) == 0:
                    split_list.append(ParticleOperator(""))
                elif oper[0] == "a":  # boson
                    split_list.append(BosonOperator(oper[1:]))
                elif oper[0] == "b":  # fermion
                    split_list.append(FermionOperator(oper[1:]))
                elif oper[0] == "d":  # antifermion
                    split_list.append(AntifermionOperator(oper[1:]))
            return split_list
        else:
            return NotImplemented

    @property
    def n_terms(self):
        return len(self)

    def partition(self) -> List:
        """Splits a ParticleOperator into a list of three terms: [fermionOps, antifermionOps, bosonOps]"""
        partitioned_list = [
            ParticleOperator(""),
            ParticleOperator(""),
            ParticleOperator(""),
        ]

        for op in self.split():
            if isinstance(op, FermionOperator):
                partitioned_list[0] *= op
            elif isinstance(op, AntifermionOperator):
                partitioned_list[1] *= op
            elif isinstance(op, BosonOperator):
                partitioned_list[2] *= op
        return partitioned_list

    def to_list(self) -> List:
        particle_op_list = []
        for oper, coeff in self.op_dict.items():
            particle_op_list.append(
                ParticleOperator({ParticleOperator.key_to_op_string(oper): coeff})
            )
        return particle_op_list

    @staticmethod
    def swap(lists_of_indices):
        n_swaps = 0
        ordered_indices = []

        for type, indices in enumerate(lists_of_indices):
            for step in range(1, len(indices)):
                key = indices[step]
                j = step - 1

                while j >= 0 and indices[j] < key:
                    indices[j + 1] = indices[j]
                    if type == 4 or type == 5:
                        n_swaps += 0
                    else:
                        n_swaps += 1
                    j = j - 1

                indices[j + 1] = key
            ordered_indices.append(indices)

        return ordered_indices, n_swaps

    @staticmethod
    def _swap_with_op_types(indices, op_types, particle_types):
        n_swaps = 0

        for step in range(1, len(indices)):
            key = indices[step]
            op_key = op_types[step]
            particle_key = particle_types[step]
            j = step - 1

            while j >= 0 and indices[j] < key:
                indices[j + 1] = indices[j]
                op_types[j + 1] = op_types[j]
                particle_types[j + 1] = particle_types[j]
                if (particle_types[j] != 2) and (particle_key != 2):
                    n_swaps += 1
                j = j - 1

            indices[j + 1] = key
            op_types[j + 1] = op_key
            particle_types[j + 1] = particle_key

        return indices, op_types, particle_types, n_swaps

    def preprocess_indices(self) -> List:
        """
        Return
        - indices_by_type: List[List]: [[fermionic creation indices], [fermionic annihilation indices],
                                        [antifermionic creation indices], [antifermionic annihilation indices],
                                        [bosonic creation indices], [bosonic annihilation indices]]
        """
        f_c_indices, f_a_indices = [], []
        af_c_indices, af_a_indices = [], []
        b_c_indices, b_a_indices = [], []

        for op in self.split():
            if op.has_fermions:
                if op.creation:
                    f_c_indices.append(op.mode)
                else:
                    f_a_indices.append(op.mode)
            elif op.has_antifermions:
                if op.creation:
                    af_c_indices.append(op.mode)
                else:
                    af_a_indices.append(op.mode)
            elif op.has_bosons:
                if op.creation:
                    b_c_indices.append(op.mode)
                else:
                    b_a_indices.append(op.mode)
        indices_by_type = [
            f_c_indices,
            f_a_indices,
            af_c_indices,
            af_a_indices,
            b_c_indices,
            b_a_indices,
        ]
        return indices_by_type

    def _order_indices(self):
        split_terms = self.preprocess_indices()
        ordered, num_swaps = self.swap(split_terms)
        coeff = (-1) ** num_swaps

        ordered_string = ""
        for i, indices in enumerate(ordered):
            if indices != []:
                if i == 0:  # bi^
                    for bi_dag_index in indices:
                        ordered_string += "b" + str(bi_dag_index) + "^ "
                elif i == 1:  # bi
                    for bi_index in indices:
                        ordered_string += "b" + str(bi_index) + " "
                elif i == 2:  # di^
                    for di_dag_index in indices:
                        ordered_string += "d" + str(di_dag_index) + "^ "
                elif i == 3:  # di
                    for di_index in indices:
                        ordered_string += "d" + str(di_index) + " "
                elif i == 4:  # ai^
                    for ai_dag_index in indices:
                        ordered_string += "a" + str(ai_dag_index) + "^ "
                elif i == 5:  # ai
                    for ai_index in indices:
                        ordered_string += "a" + str(ai_index) + " "
        return coeff * ParticleOperator(ordered_string)

    def order_indices(self) -> "ParticleOperator":

        sorted_terms = ParticleOperator()
        for term in self.to_list():
            sorted_terms += term.coeff * term._order_indices()
        return sorted_terms

    def mode_order(self) -> "ParticleOperator":
        """
        Alternative form of ordering where operators are ordered by increasing mode number, and may not be normal ordered
        Args:
            self (ParticleOperator): Operator to order by mode index
        Returns:
            (ParticleOperator): Operator ordered by mode indices
        """

        properly_ordered = ParticleOperator()

        for term, coeff in self.op_dict.items():
            particle_types = [key[0] for key in term]
            modes = [key[1] for key in term]
            op_types = [key[2] for key in term]

            new_modes, new_op_types, new_particle_types, n_swaps = (
                ParticleOperator._swap_with_op_types(modes, op_types, particle_types)
            )
            properly_ordered += ParticleOperator(
                {
                    tuple(
                        (particle_type, mode, operator_type)
                        for particle_type, mode, operator_type in zip(
                            new_particle_types, new_modes, new_op_types
                        )
                    ): (-1)
                    ** n_swaps
                    * coeff
                }
            )

        return properly_ordered

    def contains(self, operator):
        assert len(operator.to_list()) == 1
        operator_key = list(operator.op_dict.keys())[0]
        return operator_key in self.op_dict.keys()

    def group(self, atol=1e-14):
        """
        Groups terms in a sum of ParticleOperators (hermitian) into a list of term + term.dagger()
        """
        # assert self.is_hermitian

        groups = []
        used = ParticleOperator()
        operator = self.order_indices()
        for term in operator.order_indices().to_list():
            if not used.contains(term):
                if term.is_hermitian:
                    groups.append(term)
                else:
                    dag = term.dagger().normal_order().order_indices()
                    found_dag = False
                    if operator.contains(dag):
                        if np.allclose(
                            dag.coeff,
                            operator.op_dict.get(list(dag.op_dict.keys())[0], 0),
                            atol=atol,
                        ):
                            found_dag = True
                    if found_dag:
                        groups.append(term + dag)
                        used += term
                        used += dag
                    else:
                        groups.append(term)
                        used += term
        return groups

    @staticmethod
    def SB(operator, max_mode, max_occ: int):
        def _SB(operator, omega):
            fock_to_qubit = {
                (1, 0): PauliwordOp.from_list(["X", "Y"], [1 / 2, -1j / 2]),
                (0, 1): PauliwordOp.from_list(["X", "Y"], [1 / 2, 1j / 2]),
                (0, 0): PauliwordOp.from_list(["I", "Z"], [1 / 2, 1 / 2]),
                (1, 1): PauliwordOp.from_list(["I", "Z"], [1 / 2, -1 / 2]),
            }
            n_qubits = int(np.ceil(np.log2(omega + 1)))

            output = PauliwordOp.empty(n_qubits)

            for s in range(0, omega):
                if operator.all_creation:
                    ket_str = f"{s + 1:0{n_qubits}b}"
                    bra_str = f"{s:0{n_qubits}b}"
                else:
                    ket_str = f"{s:0{n_qubits}b}"
                    bra_str = f"{s + 1:0{n_qubits}b}"

                list_to_tensor = []
                for bit in range(len(ket_str)):
                    list_to_tensor.append(
                        fock_to_qubit[(int(ket_str[bit]), int(bra_str[bit]))]
                    )

                output += tensor_list(list_to_tensor) * np.sqrt(s + 1)
            return output

        assert np.log2(max_occ + 1) % 1 == 0, "Max occupancy must be given as 2^N - 1"

        n_qubits_per_mode = int(np.log2(max_occ + 1))
        n_modes = max_mode + 1
        sb = PauliwordOp.from_dictionary({"I" * (n_modes * n_qubits_per_mode): 1.0})
        if operator == ParticleOperator(""):  # Identity
            return sb
        for op in operator.split():
            mode = op.mode
            n_higher_modes = (n_modes - (mode + 1)) * n_qubits_per_mode
            n_lower_modes = mode * n_qubits_per_mode
            current_sb = tensor_list(
                [PauliwordOp.from_list(["I"], [1])] * n_higher_modes
                + [_SB(op, max_occ)]
                + [PauliwordOp.from_list(["I"], [1])] * n_lower_modes
            )
            sb *= current_sb

        return sb

    @staticmethod
    def jordan_wigner(operator, max_mode):

        def _jordan_wigner(indiv_op):
            if indiv_op.all_creation:
                return PauliwordOp.from_list(["X", "Y"], [1 / 2, -1j / 2])
            else:
                return PauliwordOp.from_list(["X", "Y"], [1 / 2, 1j / 2])

        assert not operator.has_bosons, "Must be a fermionic operator only"

        jw = PauliwordOp.from_dictionary({"I" * (max_mode + 1): 1.0})
        if operator == ParticleOperator(""):
            return jw

        for op in operator.split():
            mode = op.mode
            n_higher_modes = max_mode - mode
            n_lower_modes = mode
            current_jw = tensor_list(
                [PauliwordOp.from_list(["I"], [1])] * n_higher_modes
                + [_jordan_wigner(op)]
                + [PauliwordOp.from_list(["Z"], [1])] * n_lower_modes
            )
            jw *= current_jw

        return jw

    @staticmethod
    def allocate_qubits(
        max_fermionic_mode: int = None,
        max_antifermionic_mode: int = None,
        max_bosonic_mode: int = None,
        max_bosonic_occupancy: int = None,
        return_details: bool = False,
    ):
        if return_details:
            qubits = []
            n_fermionic_qubits = 0
            n_antifermionic_qubits = 0
            n_bosonic_qubits = 0
            if max_fermionic_mode is not None:
                qubits.append(max_fermionic_mode + 1)
                n_fermionic_qubits = max_fermionic_mode + 1
            if max_antifermionic_mode is not None:
                qubits.append(max_antifermionic_mode + 1)
                n_antifermionic_qubits = max_antifermionic_mode + 1
            if max_bosonic_mode is not None:
                qubits.append(
                    (max_bosonic_mode + 1)
                    * int(np.ceil(np.log2(max_bosonic_occupancy + 1)))
                )
                n_bosonic_qubits = (max_bosonic_mode + 1) * int(
                    np.ceil(np.log2(max_bosonic_occupancy + 1))
                )
            return qubits, n_fermionic_qubits, n_antifermionic_qubits, n_bosonic_qubits
        else:
            qubits = []
            if max_fermionic_mode is not None:
                qubits.append(max_fermionic_mode + 1)
            if max_antifermionic_mode is not None:
                qubits.append(max_antifermionic_mode + 1)
            if max_bosonic_mode is not None:
                qubits.append(
                    (max_bosonic_mode + 1)
                    * int(np.ceil(np.log2(max_bosonic_occupancy + 1)))
                )

        return qubits

    def to_paulis(
        self,
        max_fermionic_mode: int = None,
        max_antifermionic_mode: int = None,
        max_bosonic_mode: int = None,
        max_bosonic_occupancy: int = None,
        zero_threshold: float = 1e-15,
    ) -> PauliwordOp:

        qubits = self.allocate_qubits(
            max_fermionic_mode=max_fermionic_mode,
            max_antifermionic_mode=max_antifermionic_mode,
            max_bosonic_mode=max_bosonic_mode,
            max_bosonic_occupancy=max_bosonic_occupancy,
        )
        pauli_op = PauliwordOp.empty(n_qubits=sum(qubits))

        for term in self.to_list():
            mapped_term = []
            partitioned_op = term.partition()
            coeff_of_term = term.coeff
            f_op = partitioned_op[0]
            af_op = partitioned_op[1]
            b_op = partitioned_op[2]
            if max_fermionic_mode is not None:
                mapped_term.append(
                    self.jordan_wigner(f_op, max_mode=max_fermionic_mode)
                )
            if max_antifermionic_mode is not None:
                mapped_term.append(
                    self.jordan_wigner(af_op, max_mode=max_antifermionic_mode)
                )
                if f_op != ParticleOperator("") and af_op != ParticleOperator(
                    ""
                ):  # If there are fermions, multiply fermion qubits by Z for parity
                    for _ in range(len(af_op.split())):
                        mapped_term[0] *= PauliwordOp.from_list(
                            ["Z" * (max_fermionic_mode + 1)], [1]
                        )
            if max_bosonic_mode is not None:
                mapped_term.append(
                    self.SB(
                        b_op, max_mode=max_bosonic_mode, max_occ=max_bosonic_occupancy
                    )
                )

            pauli_op += tensor_list(mapped_term) * coeff_of_term

        return pauli_op.cleanup(zero_threshold=zero_threshold)

    def parse(self) -> List:
        """Merges neighboring creation/annihilation operators together into NumberOperators when acting on the same modes.

        Merging only occurs when annihilation operators are immediately followed by creation operators. Therefore, there is
            a preference for normal ordering the terms before calling this function.
        """
        if len(list(self.op_dict.keys())) != 1:
            return NotImplemented

        # ops = list(self.op_dict.keys())[0].split(" ")
        ops = ParticleOperator.key_to_op_string(list(self.op_dict.keys())[0]).split(" ")

        split_list = []
        i = 0
        stack = []

        def _empty_stack(stack):
            operators = []
            while len(stack) > 0:
                current_operator_string = str(stack[0][1])
                if stack[0][2]:
                    current_operator_string += "^"

                if stack[0][0] == "a":
                    operators.append(BosonOperator(current_operator_string))
                elif stack[0][0] == "b":
                    operators.append(FermionOperator(current_operator_string))
                elif stack[0][0] == "d":
                    operators.append(AntifermionOperator(current_operator_string))
                else:
                    raise RuntimeError("particle type unknown: {}".format(stack[0]))
                stack = stack[1:]
            return operators

        def _get_operator_tuple(index):
            oper = ops[index]
            particle_type = oper[0]
            oper = oper[1:]
            creation = False
            if oper[-1] == "^":
                creation = True
                oper = oper[:-1]
            mode = int(oper)
            return (particle_type, mode, creation)

        while i < len(ops):
            operator = _get_operator_tuple(i)
            if len(stack) == 0:
                if operator[2]:
                    stack.append(operator)
                else:
                    split_list += _empty_stack([operator])
            else:
                if (operator[0] == stack[-1][0]) and (operator[1] == stack[-1][1]):
                    if operator[2]:
                        stack.append(operator)
                    else:
                        power = 1
                        while (i + power < len(ops)) and (power < len(stack)):
                            next_operator = _get_operator_tuple(i + power)
                            if (
                                (next_operator[0] == stack[-1][0])
                                and (next_operator[1] == stack[-1][1])
                                and (not next_operator[2])
                            ):
                                power += 1
                            else:
                                break

                        split_list += _empty_stack(stack[:-power])
                        stack = []
                        split_list.append(
                            OccupationOperator(operator[0], operator[1], power)
                        )
                        i += power - 1
                else:
                    split_list += _empty_stack(stack)
                    stack = [operator]
            i += 1
        split_list += _empty_stack(stack)
        return split_list

    def dagger(self) -> "ParticleOperator":
        dagger_dict = {}

        for i, coeff_i in self.op_dict.items():
            if i == ():  # I^dagger = I
                dagger_dict[()] = coeff_i.conjugate()
            else:
                key = []
                for oper in i[::-1]:
                    key.append((oper[0], oper[1], 1 - oper[2]))
                dagger_dict[tuple(key)] = coeff_i.conjugate()

        return ParticleOperator(dagger_dict)

    def __rmul__(self, other):
        coeffs = np.array(list(self.op_dict.values()))
        return ParticleOperator(dict(zip(self.op_dict.keys(), other * coeffs)))

    def __mul__(self, other):
        if isinstance(other, ParticleOperator):
            product_dict = {}
            for op1, coeffs1 in list(self.op_dict.items()):
                for op2, coeffs2 in list(other.op_dict.items()):
                    if op1 == () and op2 == ():
                        product_dict[()] = coeffs1 * coeffs2
                    else:
                        # Add .strip() to remove trailing spaces when multipying with identity (treated as ' ')
                        product_dict[op1 + op2] = coeffs1 * coeffs2
            return ParticleOperator(product_dict)

    def __pow__(self, other) -> "ParticleOperator":
        if other == 0:
            return ParticleOperator(" ")
        else:
            op = self
            for _ in range(1, other):
                op *= self
            return op

    def __sub__(self, other):
        coeffs = list(other.op_dict.values())
        neg_other = ParticleOperator(
            dict(zip(other.op_dict.keys(), -1 * np.array(coeffs)))
        )
        return self + neg_other

    def __len__(self):
        return len(self.op_dict)

    def __eq__(self, other):
        return self.op_dict == other.op_dict

    def __neq__(self, other):
        return self.op_dict != other.op_dict

    @property
    def coeff(self):
        assert (
            len(self) == 1
        ), "coeff property exists for 1 element in op_dict only. Use .coeffs property"
        return next(iter(self.op_dict.values()))

    @property
    def coeffs(self):
        return list(self.op_dict.values())

    @property
    def has_fermions(self):
        for key in self.op_dict.keys():
            if "b" in ParticleOperator.key_to_op_string(key):
                return True
        return False

    @property
    def has_antifermions(self):
        for key in self.op_dict.keys():
            if "d" in ParticleOperator.key_to_op_string(key):
                return True
        return False

    @property
    def has_bosons(self):
        for key in self.op_dict.keys():
            if "a" in ParticleOperator.key_to_op_string(key):
                return True
        return False

    @property
    def n_bosons(self):
        for key in self.op_dict.keys():
            if "a" in ParticleOperator.key_to_op_string(key):
                return ParticleOperator.key_to_op_string(key).count("a")
            else:
                return 0

    @property
    def n_fermions(self):
        for key in self.op_dict.keys():
            if "b" in ParticleOperator.key_to_op_string(key):
                return ParticleOperator.key_to_op_string(key).count("b")
            else:
                return 0

    @property
    def n_antifermions(self):
        for key in self.op_dict.keys():
            if "d" in ParticleOperator.key_to_op_string(key):
                return ParticleOperator.key_to_op_string(key).count("d")
            else:
                return 0

    def normal_order(self) -> "ParticleOperator":
        """
        # Returns a new ParticleOperator object with a normal ordered hash table
        # normal ordering: b^dagger before b; d^dagger before d; a^dagger before a
        # b2 b1^ a0 b3 -> b1^ b2 b3 a0
        """
        if list(self.op_dict.keys())[0] == ():
            return self
        po_list = self.to_list()
        output_op = {}

        # for each PO in POS, convert it to normal order
        for particle_op in po_list:
            coeff_op = list(particle_op.op_dict.values())[0]
            normal_ordered_op = particle_op._normal_order(coeff_op)

            # cleaning up the 0 coeffs when outputting the result
            for key, val in normal_ordered_op.items():
                final_val = val + output_op.get(key, 0)
                if final_val == 0:
                    if key in output_op:
                        del output_op[key]
                else:
                    output_op[key] = final_val
        return ParticleOperator(output_op)

    def split_to_string(self):
        """
        this function splits a particle operator into 3 lists of different types
        of operators (represented by string)
        """
        fermion_list = []
        antifermion_list = []
        boson_list = []
        swap_bd = 0
        for op in ParticleOperator.key_to_op_string(list(self.op_dict.keys())[0]).split(
            " "
        ):
            if op[0] == "a":  # boson
                boson_list.append(op)
            elif op[0] == "b":  # fermion
                swap_bd += len(antifermion_list)
                fermion_list.append(op)
            elif op[0] == "d":  # antifermion
                antifermion_list.append(op)
            else:
                raise Exception("Unknown particle type appear in normal order")

        return fermion_list, antifermion_list, boson_list, swap_bd

    def is_canonical_ordered(self):
        # this method is meant to check particle operator with only one term
        assert len(self.op_dict) == 1

        # parse the op_str into bs, ds, and as
        fermion_list, antifermion_list, boson_list, _ = self.split_to_string()
        return (
            self.is_normal_ordered(fermion_list)
            and self.is_normal_ordered(antifermion_list)
            and self.is_normal_ordered(boson_list)
        )

    def _normal_order(self, coeff) -> dict:
        """
        # inner func only operates on one particle operator instance
        # i.e. ParticleOperator("a0 b0 a2^ b0^ b0^ b1 d3 a2^ d3^")
        """
        # prevent normal ordering identity/empty op
        if list(self.op_dict.keys())[0] == ():
            return self.op_dict

        if ParticleOperator.has_duplicates(
            ParticleOperator.key_to_op_string(next(iter(self.op_dict)))
        ):
            return {}

        # parse the op_str into bs, ds, and as and normal ordering them separately
        fermion_list, antifermion_list, boson_list, swap_bd = self.split_to_string()

        # insertion sort the ops
        # note that extra terms will be added for nontrivial swaps
        # normal_{x}s is a particle operator that contains the key with particle op only in type x
        normal_bs = self.insertion_sort(
            fermion_list, 0 if len(fermion_list) == 0 else 1
        )
        if "" in normal_bs:
            normal_bs["I"] = normal_bs.pop("")

        normal_ds = self.insertion_sort(
            antifermion_list, 0 if len(antifermion_list) == 0 else 1
        )
        if "" in normal_ds:
            normal_ds["I"] = normal_ds.pop("")

        normal_as = self.insertion_sort(boson_list, 0 if len(boson_list) == 0 else 1)
        if "" in normal_as:
            normal_as["I"] = normal_as.pop("")

        # sign change between swapping bs and ds
        if (swap_bd % 2) == 0:
            sign_swap_bd = 1
        else:
            sign_swap_bd = -1

        output_dict = {}

        # Generate combinations of keys, considering empty dictionaries
        b_keys = normal_bs.keys() if normal_bs else [""]
        d_keys = normal_ds.keys() if normal_ds else [""]
        a_keys = normal_as.keys() if normal_as else [""]

        if not (normal_bs and normal_ds):
            sign_swap_bd = 1

        for b_key, d_key, a_key in product(b_keys, d_keys, a_keys):
            if b_key or d_key or a_key:  # Ensure at least one key is non-empty

                b_key_new = "" if b_key == "I" else b_key
                d_key_new = "" if d_key == "I" else d_key
                a_key_new = "" if a_key == "I" else a_key

                keys = [b_key_new, d_key_new, a_key_new]
                new_key = " ".join(key for key in keys if key)

                new_value = (
                    normal_bs.get(b_key, 1)
                    * normal_ds.get(d_key, 1)
                    * normal_as.get(a_key, 1)
                ) * sign_swap_bd

                output_dict[new_key] = new_value * coeff

        return output_dict

    @staticmethod
    def has_duplicates(op):
        """
        used for fermions/antifermions
        checks if there are two identical particle operators acting on the same
        mode existing in a row (ignoring the in between ops operatoring on diff
        modes)
        i.e. b0 b1 b2 b0 -> True == has duplicates -> kill the state
             b0 b0^ b0 -> False == not has duplicates
        """
        # Split the expression into individual terms
        terms = op.split()

        # Dictionary to store the terms we've seen
        seen_terms = {}

        for term in terms:
            # Extract the letter, mode, and check for '^'
            match = re.match(r"([a-zA-Z])(\d+)(\^?)", term)

            if match:
                letter = match.group(1)
                mode = match.group(2)
                create_or_annihilate = bool(match.group(3))

                if letter == "a":
                    continue

                # Create a key using letter and mode to check duplicates
                key = (letter, mode)

                # If we've seen this key before and both have '^', return 0
                if key in seen_terms and seen_terms[key] == create_or_annihilate:
                    return True

                # Store the presence or absence of '^' for this term
                seen_terms[key] = create_or_annihilate

        # If no consecutive terms found with same letter, mode, and '^', return 1
        return False

    def is_normal_ordered(self, ops):
        """
        check if the ops is normal_ordered
        expects: ops should be a list of ops in a single type
        """
        found_non_creation = False
        for op in ops:
            # if op is creation
            if op[-1] == "^":
                if found_non_creation:
                    return False
            else:
                found_non_creation = True
        return True

    def insertion_sort(self, ops, coeff) -> dict:
        """
        main function handles normal ordering
        """
        # base case(s)
        if not ops or coeff == 0:
            return {}

        result_dict = {}

        if self.is_normal_ordered(ops):
            # return the op with normal ordered key
            if len(ops) != 0:
                result_str = " ".join(ops)
                result_dict[result_str] = coeff
            return result_dict

        # recursive case
        else:
            # if it's a list of boson operators
            if ops[0][0] == "a":
                type_coeff = 1
                is_boson = True
            else:
                type_coeff = -1
                is_boson = False
            for i in range(1, len(ops)):
                right = ops[i]
                j = i - 1

                # kill the term if it's two consecutive and identical ops
                if not is_boson and ops[j] == right:
                    return {}

                while (
                    j >= 0
                    and not ops[j][-1] == "^"
                    and not (not ops[j][-1] == "^" and not right[-1] == "^")
                ):
                    # case for non-trivial swap if two ops share the same mode
                    # --> introduce extra terms
                    # ops[j] is the left annihilation operator

                    if ops[j] == right[:-1]:
                        # for bosons: a_i a_i^ = 1 + a_i^ a_i
                        # for (anti)fermions: b_i b_i^ = 1 - b_i^ b_i (same for 'd's)

                        new_op_str = right + " " + ops[j]

                        left_str = " ".join(ops[:j])
                        right_str = " ".join(ops[j + 2 : len(ops)])

                        identity_term = (left_str + " " + right_str).strip()
                        other_term = (
                            left_str + " " + new_op_str + " " + right_str
                        ).strip()

                        result_dict[identity_term] = coeff + result_dict.get(
                            identity_term, 0
                        )
                        result_dict[other_term] = (
                            coeff + result_dict.get(other_term, 0)
                        ) * type_coeff

                        output_dict = {}
                        for key_str, val_coeff in result_dict.items():
                            if key_str != "":
                                recur_ops = key_str.split(" ")
                                sub_result_dict = self.insertion_sort(
                                    recur_ops, val_coeff
                                )
                                for key, value in sub_result_dict.items():
                                    if value != 0:
                                        output_dict[key] = (
                                            output_dict.get(key, 0) + value
                                        )
                            else:
                                output_dict[key_str] = (
                                    output_dict.get(key_str, 0) + val_coeff
                                )

                        return output_dict

                    # for bs and ds, even modes are different, when we swap them,
                    # we need to consider the case to change the sign
                    if not is_boson:
                        coeff *= -1

                    # swap the ops
                    ops[j + 1] = ops[j]
                    j -= 1
                ops[j + 1] = right

            # return the op with normal ordered key
            if len(ops) != 0:
                result_str = " ".join(ops)
                result_dict[result_str] = coeff

            # return result_op
            return result_dict

    @property
    def max_mode(self):
        maximum_mode = None
        for outer_tuple in self.op_dict.keys():
            for inner_tuple in outer_tuple:
                mode = inner_tuple[1]  # Extract the middle entry
                if maximum_mode is None or mode > maximum_mode:
                    maximum_mode = mode
        return maximum_mode

    @property
    def modes(self):
        modes_list = []

        for term in self.op_dict:
            modes_list.append([key[1] for key in term])

        return modes_list

    @property
    def max_fermionic_mode(self):
        if not self.has_fermions:
            return None

        max_fermionic_mode = 0

        for op_product in self.to_list():
            for op in op_product.split():
                if isinstance(op, FermionOperator):
                    if op.mode > max_fermionic_mode:
                        max_fermionic_mode = op.mode
        return max_fermionic_mode

    @property
    def max_antifermionic_mode(self):
        if not self.has_antifermions:
            return None

        max_antifermionic_mode = 0

        for op_product in self.to_list():
            for op in op_product.split():
                if isinstance(op, AntifermionOperator):
                    if op.mode > max_antifermionic_mode:
                        max_antifermionic_mode = op.mode
        return max_antifermionic_mode

    @property
    def max_bosonic_mode(self):
        if not self.has_bosons:
            return None

        max_bosonic_mode = 0

        for op_product in self.to_list():
            for op in op_product.split():
                if isinstance(op, BosonOperator):
                    if op.mode > max_bosonic_mode:
                        max_bosonic_mode = op.mode
        return max_bosonic_mode

    def commutator(self, other: "ParticleOperator") -> "ParticleOperator":
        # [A, B] = AB - BA
        return (
            (self * other).normal_order() - (other * self).normal_order()
        )._cleanup()

    def anticommutator(self, other: "ParticleOperator") -> "ParticleOperator":
        # {A, B} = AB + BA
        return (
            (self * other).normal_order() + (other * self).normal_order()
        )._cleanup()

    @staticmethod
    def random(
        particle_types: Optional[
            List[Literal["fermion", "antifermion", "boson"]]
        ] = None,
        n_terms: int = None,
        max_mode: int = None,
        max_len_of_terms: int = None,
        complex_coeffs: bool = True,
        normal_order: bool = True,
    ):
        # Assign None parameters
        if n_terms is None:
            n_terms = np.random.randint(1, 20)
        if max_mode is None:
            max_mode = np.random.randint(1, 20)
        if max_len_of_terms is None:
            max_len_of_terms = np.random.randint(1, 4)
        if particle_types is None:
            particle_types = ["fermion", "antifermion", "boson"]

        random_op_dict = {}
        input_map = {"fermion": "b", "antifermion": "d", "boson": "a"}
        op_types = [input_map[ptype] for ptype in particle_types]
        c_or_a = ["", "^"]

        for i in range(n_terms):
            len_current_term = np.random.randint(1, max_len_of_terms + 1)
            current_term = ""
            for j in range(len_current_term):
                particle = np.random.choice(op_types)
                mode = np.random.randint(0, max_mode)
                creation = np.random.choice(c_or_a)
                operator = str(particle) + str(mode) + str(creation) + " "
                if (particle == "a") or (operator not in current_term):
                    current_term += operator
            random_op_dict[current_term[:-1]] = np.random.uniform(
                -100, 100
            ) + 1j * np.random.uniform(-100, 100) * int(
                complex_coeffs
            )  # a + ib

        if normal_order:
            return ParticleOperator(random_op_dict).normal_order()
        else:
            return ParticleOperator(random_op_dict)

    def remove_identity(self):
        if () in self.op_dict:
            del self.op_dict[()]

    def remove_antifermions(self):
        no_antifermion_op = {}
        for term in self:
            if not term.has_antifermions:
                no_antifermion_op.update(term.op_dict)

        return ParticleOperator(no_antifermion_op)

    def pop_identity(self):
        if () in self.op_dict:
            id_coeff = self.op_dict[()]
            del self.op_dict[()]
            return id_coeff

    @staticmethod
    def fermionic_parity(fermi_occ):
        # bubble sort (anti)fermionic occupancies and count swaps to induce parity
        n = len(fermi_occ)
        swap_count = 0
        for i in range(n):
            for j in range(0, n - i - 1):
                if fermi_occ[j] > fermi_occ[j + 1]:
                    # Swap elements
                    fermi_occ[j], fermi_occ[j + 1] = fermi_occ[j + 1], fermi_occ[j]
                    swap_count += 1
        return fermi_occ, (-1) ** swap_count

    @staticmethod
    def partition_term(term_key):
        # Splits a ParticleOperator by particle type
        grouped_ops = defaultdict(list)
        for t in term_key:
            grouped_ops[t[0]].append(t[1])  # append mode acted on
        return dict(grouped_ops)

    def act_on_vacuum(self):

        if self.op_dict == {}:  # empty op
            return 0
        elif self.op_dict == {(): 1.0}:  # Identity|v⟩ = |v⟩
            return Fock.vacuum()

        state_dict = {}
        for key, val in self.normal_order().op_dict.items():
            if not ParticleOperator(
                {key: val}
            ).all_creation:  # If we normal order the key and there is an annihilation operator, its action on |v⟩ will return 0
                continue
            split_key = ParticleOperator.partition_term(key)
            if 0 in split_key:
                f_occ, f_parity = ParticleOperator.fermionic_parity(split_key[0])
                if len(f_occ) != len(set(f_occ)):  # i.e. there are duplicate modes
                    continue
            else:
                f_occ = []
                f_parity = 1
            if 1 in split_key:
                af_occ, af_parity = ParticleOperator.fermionic_parity(split_key[1])
                if len(af_occ) != len(set(af_occ)):  # i.e. there are duplicate modes
                    continue
            else:
                af_occ = []
                af_parity = 1
            if 2 in split_key:
                b_occ = [(num, count) for num, count in Counter(split_key[2]).items()]
                b_coeff = np.prod(
                    np.sqrt(scipy.special.factorial(np.array(b_occ)[:, 1]))
                )
            else:
                b_occ = []
                b_coeff = 1

            coeff = val * f_parity * af_parity * b_coeff
            state_dict_key = (
                tuple(sorted(f_occ)),
                tuple(sorted(af_occ)),
                tuple(sorted(b_occ)),
            )
            state_dict[state_dict_key] = coeff + state_dict.get(state_dict_key, 0)

        if state_dict == {}:
            return 0
        else:
            return Fock(state_dict=state_dict)

    @property
    def all_creation(self):
        # Return true if all operators in a ParticleOperator are creation operators
        tup = []
        for op in self.op_dict:
            tup += op
            creation = [t[-1] for t in tup]
        return all(x == 1 for x in creation)

    def all_annihilation(self):
        # Takes in the tuple key of one ParticleOperator term and returns True if all ops are annihilation
        assert len(self.op_dict) == 1
        tup = next(iter(self.op_dict))
        annihilation = [t[-1] for t in tup]
        return all(x == 0 for x in annihilation)

    def VEV(self):
        """
        VEV: Vacuum mn_bosons_in_opatrix element
        Calculate ⟨0|operator|0⟩ by normal ordering the operator,
          and taking the coefficient of the identity term (if no identity term, return 0)
        """
        return self.normal_order().op_dict.get((), 0)

    @property
    def is_hermitian(self):
        return self.op_dict == self.dagger().order_indices().op_dict

    def has_identity(self):
        return () in self.op_dict.keys()


class Fock(ParticleOperator):

    def __init__(
        self,
        f_occ: List = None,
        af_occ: List = None,
        b_occ: List = None,
        state_dict: Dict[Tuple[Tuple, Tuple, Tuple], complex] = None,
        coeff: complex = 1.0,
    ):
        if state_dict is not None:
            self.state_dict = state_dict

            op_dict = {}
            for key, val in self.state_dict.items():
                f_occ = list(key[0])
                af_occ = list(key[1])
                b_occ = list(key[2])

                f_tup = tuple([(0, i, 1) for i in f_occ])
                af_tup = tuple([(1, i, 1) for i in af_occ])
                b_tup = tuple((2, i, 1) for i, b in b_occ for _ in range(b))

                current_key = f_tup + af_tup + b_tup

                if b_occ != []:
                    b_coeff = np.prod(
                        np.sqrt(scipy.special.factorial(np.array(b_occ)[:, 1]))
                    )
                else:
                    b_coeff = 1

                op_dict[current_key] = val / b_coeff

            self.state_opdict = op_dict
            super().__init__(
                op_dict,
            )

        else:
            self.f_occ = f_occ
            self.af_occ = af_occ
            self.b_occ = b_occ
            # Ensure Pauli exclusion and clean up b_occ:
            b_modes = [x[0] for x in b_occ]
            if len(b_modes) != len(set(b_modes)):
                combined = defaultdict(int)
                for key, value in b_occ:
                    combined[key] += value
                b_occ = list(combined.items())

            assert len(f_occ) == len(set(f_occ)) and len(af_occ) == len(
                set(af_occ)
            ), "This state violates the Pauli exclusion principle"

            self.state_dict = {
                (
                    tuple(sorted(f_occ)),
                    tuple(sorted(af_occ)),
                    tuple(sorted(tuple([(n, m) for (n, m) in b_occ if m != 0]))),
                ): coeff
            }
            f_tup = tuple([(0, i, 1) for i in f_occ])
            af_tup = tuple([(1, i, 1) for i in af_occ])
            b_tup = tuple((2, i, 1) for i, b in b_occ for _ in range(b))

            if b_occ != []:
                b_coeff = np.prod(
                    np.sqrt(scipy.special.factorial(np.array(b_occ)[:, 1]))
                )
            else:
                b_coeff = 1

            key = f_tup + af_tup + b_tup
            self.state_opdict = {key: coeff / b_coeff}

            super().__init__(self.state_opdict, coeff=coeff / b_coeff)

    def __str__(self):
        if len(self.state_dict) == 0:
            return "0"
        else:
            output_str = ""
            for state, coeff in self.state_dict.items():
                output_str += f"{coeff} * |{state}⟩ +"
                output_str += "\n"
            return output_str[:-3]

    def __repr__(self) -> str:
        return self.__str__()

    def display(self):
        display(Latex("$" + self.__str__() + "$"))

    @property
    def coeff(self):
        return next(iter(self.state_dict.values()))

    def __rmul__(self, other):
        if isinstance(other, ParticleOperator):
            # TODO: pop term then add back into other (for now just copying)
            op_dict_copy = deepcopy(other.op_dict)
            const = op_dict_copy.pop((), Fock([], [], [], coeff=0))  # Identity

            output = ParticleOperator({})

            new_op = ParticleOperator(op_dict_copy) * ParticleOperator(self.op_dict)
            if new_op.op_dict != {}:  # i.e. has more than just the identity
                for term in new_op.normal_order().to_list():
                    if term.all_creation:
                        output += term
            else:
                return const * self
            identity_term = const * self  # Already a Fock state
            other = output.act_on_vacuum()  # Turn PO -> Fock
            if identity_term.coeff != 0:
                return identity_term + other
            else:
                return other

        elif isinstance(other, (float, int, complex)):
            if other == 0:
                return 0
            new_dict = {
                key: other * val
                for key in self.op_dict.keys()
                for val in self.op_dict.values()
            }
            return ParticleOperator(new_dict).act_on_vacuum()

    def __add__(self, other):
        new_dict = defaultdict(float, self.state_dict)
        for key, value in other.state_dict.items():
            new_dict[key] += value
        out_state = Fock(state_dict=new_dict)
        if len(out_state.state_dict) == 0:
            return 0
        else:
            return out_state

    @staticmethod
    def vacuum():
        return Fock([], [], [])

    @staticmethod
    def fermion_fock_to_qubit(occ, max_mode: int = None):
        if occ != []:
            fock_list = occ
            if max_mode is not None:
                qubit_state = [0] * (max_mode + 1)
            else:
                qubit_state = [0] * (max(occ) + 1)

            for index in fock_list:
                qubit_state[index] = 1
            return qubit_state[::-1]

        else:
            if max_mode is not None:
                return [0] * (max_mode + 1)
            else:
                return []

    @staticmethod
    def boson_fock_to_qubit(b_occ, max_bosonic_occupancy: int, max_mode: int = None):
        if b_occ != []:
            if max_mode is not None:
                _max = max_mode
            else:
                _max = max(b_occ, key=lambda x: x[0])[0]
            n_qubits_one_mode = int(np.log2(max_bosonic_occupancy + 1))
            qubit_occ = ["0" * n_qubits_one_mode] * (_max + 1)
            for occ in b_occ:
                qubit_occ[_max - occ[0]] = f"{occ[1]:0{n_qubits_one_mode}b}"

            return [int(char) for segment in qubit_occ for char in segment]

        else:
            if max_mode is not None:
                return [0] * (max_mode + 1) * int(np.log2(max_bosonic_occupancy + 1))
            else:
                return []

    def to_qubit_state(
        self,
        max_bosonic_occupancy: int = None,
        max_fermionic_mode: int = None,
        max_antifermionic_mode: int = None,
        max_bosonic_mode: int = None,
    ):
        qubits, nfq, nafq, nbq = self.allocate_qubits(
            max_fermionic_mode=max_fermionic_mode,
            max_antifermionic_mode=max_antifermionic_mode,
            max_bosonic_mode=max_bosonic_mode,
            max_bosonic_occupancy=max_bosonic_occupancy,
            return_details=True,
        )

        total_qubit_state = QuantumState([0] * sum(qubits)) * 0

        for state_key, coeff in self.state_dict.items():
            state = [0] * sum(qubits)
            fock = Fock(state_dict={state_key: coeff})

            state_dict_form = list(state_key)
            if fock.has_fermions:
                state[:(nfq)] = fock.fermion_fock_to_qubit(
                    list(state_dict_form[0]), max_mode=max_fermionic_mode
                )
            if fock.has_antifermions:
                state[(nfq) : (nfq) + (nafq)] = fock.fermion_fock_to_qubit(
                    list(state_dict_form[1]), max_mode=max_antifermionic_mode
                )
            if fock.has_bosons:
                state[(nfq) + (nafq) :] = fock.boson_fock_to_qubit(
                    list(state_dict_form[2]),
                    max_bosonic_occupancy=max_bosonic_occupancy,
                    max_mode=max_bosonic_mode,
                )
            total_qubit_state += QuantumState(state) * coeff
        return total_qubit_state


class FermionOperator(ParticleOperator):

    def __init__(self, input_string):
        if input_string[-1] == "^":
            self.mode = int(input_string[:-1])
            self.creation = True
        else:
            self.mode = int(input_string)
            self.creation = False

        self.particle_type = "fermion"
        super().__init__("b" + input_string)

    def __str__(self):
        return super().__str__()

    def __mul__(self, other):
        return super().__mul__(other)

    def __add__(self, other):
        return super().__add__(other)

    def _operate_on_state(self, other) -> "Fock":
        f_occ = list(list(other.state_dict.keys())[0][0])

        if self.creation and self.mode not in f_occ:
            f_occ.append(self.mode)
            f_occ = sorted(f_occ)
            # Get parity
            coeff = (-1) ** len(f_occ[: f_occ.index(self.mode)])
        elif not self.creation and self.mode in f_occ:
            # Get parity
            coeff = (-1) ** len(f_occ[: f_occ.index(self.mode)])
            f_occ.remove(self.mode)
            f_occ = sorted(f_occ)
        else:
            return Fock([], [], []), 0

        return (
            Fock(
                f_occ=sorted(f_occ),
                af_occ=list(list(other.state_dict.keys())[0][1]),
                b_occ=list(list(other.state_dict.keys())[0][2]),
            ),
            coeff,
        )


class AntifermionOperator(ParticleOperator):
    def __init__(self, input_string):
        if input_string[-1] == "^":
            self.mode = int(input_string[:-1])
            self.creation = True
        else:
            self.mode = int(input_string)
            self.creation = False

        self.particle_type = "antifermion"
        super().__init__("d" + input_string)

    def __str__(self):
        return super().__str__()

    def __mul__(self, other):
        return super().__mul__(other)

    def __add__(self, other):
        return super().__add__(other)

    def _operate_on_state(self, other) -> "Fock":
        af_occ = list(list(other.state_dict.keys())[0][1])
        if self.creation and self.mode not in af_occ:
            af_occ.append(self.mode)
            # Get parity
            af_occ = sorted(af_occ)
            coeff = (-1) ** len(af_occ[: af_occ.index(self.mode)])
        elif not self.creation and self.mode in af_occ:
            # Get parity
            coeff = (-1) ** len(af_occ[: af_occ.index(self.mode)])
            af_occ.remove(self.mode)
            af_occ = sorted(af_occ)
        else:
            return Fock([], [], []), 0

        return (
            Fock(
                f_occ=list(list(other.state_dict.keys())[0][0]),
                af_occ=sorted(af_occ),
                b_occ=list(list(other.state_dict.keys())[0][2]),
            ),
            coeff,
        )


class BosonOperator(ParticleOperator):
    def __init__(self, input_string):
        if input_string[-1] == "^":
            self.mode = int(input_string[:-1])
            self.creation = True
        else:
            self.mode = int(input_string)
            self.creation = False

        self.particle_type = "boson"
        super().__init__("a" + input_string)

    def __str__(self):
        return super().__str__()

    def __mul__(self, other):
        return super().__mul__(other)

    def __add__(self, other):
        return super().__add__(other)

    def _operate_on_state(self, other) -> "Fock":
        b_occ = list(list(other.state_dict.keys())[0][2])
        state_modes, state_occupancies = [i[0] for i in b_occ], [i[1] for i in b_occ]

        if self.creation and self.mode in state_modes:
            index = state_modes.index(self.mode)
            state_occupancies[index] += 1
            coeff = np.sqrt(state_occupancies[index])  # sqrt(n + 1)
        elif self.creation and self.mode not in state_modes:
            state_modes.append(self.mode)
            state_occupancies.append(1)
            coeff = 1
        elif not self.creation and self.mode in state_modes:
            index = state_modes.index(self.mode)
            state_occupancies[index] -= 1
            coeff = np.sqrt(state_occupancies[index] + 1)  # sqrt(n)
        else:
            return Fock([], [], []), 0

        updated_bos_state = list(zip(state_modes, state_occupancies))
        sorted_updated_bos_state = sorted(updated_bos_state, key=lambda x: x[0])

        if len(b_occ) == 0:
            coeff = 1
        return (
            Fock(
                f_occ=list(list(other.state_dict.keys())[0][0]),
                af_occ=list(list(other.state_dict.keys())[0][1]),
                b_occ=sorted_updated_bos_state,
            ),
            coeff,
        )


class OccupationOperator(ParticleOperator):
    def __init__(self, particle_type, mode, power):
        self.particle_type = particle_type
        self.mode = int(mode)
        self.power = int(power)

        if (self.particle_type != "a") and (power != 1):
            raise ValueError(
                "power of OccupationOperator can only be greater than 1 for bosonic ladder operators"
            )

        creation_op = ParticleOperator(self.particle_type + str(self.mode) + "^")
        annihilation_op = ParticleOperator(self.particle_type + str(self.mode))
        occupation_op = (creation_op**power) * (annihilation_op**power)
        super().__init__(occupation_op.op_dict)


class NumberOperator(OccupationOperator):
    def __init__(self, particle_type, mode):
        super().__init__(particle_type, mode, 1)

import numpy as np
from sympy import *
from typing import List, Union, Optional, Dict, Tuple
from IPython.display import display, Latex
from collections import defaultdict
import re
from copy import deepcopy


class Fock:
    """
    Defines a Fock state (ket) of the form |f, \bar{f}, b⟩

    Parameters:
    ferm_occupancy, antiferm_occupancy, bos_occupancy: List
    e.g. a hadron with a fermion in mode 2, an antifermion in mode 1 and 2 bosons
    in mode 1 is
    ferm_occupancy = [2]
    antiferm_occupancy = [1]
    bos_occupancy = [(1, 2)]

    """

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

        else:
            self.state_dict = {
                (
                    tuple(f_occ),
                    tuple(af_occ),
                    tuple([(n, m) for (n, m) in b_occ if m != 0]),
                ): coeff
            }

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

    def to_list(self) -> List:
        state_list = []
        for state, coeff in self.state_dict.items():
            state_list.append(Fock(state_dict={state: coeff}))
        return state_list

    def __rmul__(self, other):

        if isinstance(other, (float, int, complex)):
            coeffs = np.array(list(self.state_dict.values()))
            return Fock(state_dict=dict(zip(self.state_dict.keys(), other * coeffs)))

        elif isinstance(other, Fock):
            raise Exception("Cannot multiply a ket to the left of a ket")

        elif isinstance(other, ParticleOperator):
            output_state_dict = {}
            for op in other.to_list():
                op_coeff = next(iter(op.op_dict.values()))
                for state in self.to_list():
                    state_coeff = next(iter(state.state_dict.values()))
                    split_coeff = 1
                    if next(iter(op.op_dict)) == " ":  # Identity * |state> = |state>
                        output_state_dict[next(iter(state.state_dict))] = (
                            op_coeff * state_coeff
                        ) + output_state_dict.get(next(iter(state.state_dict)), 0)
                    else:
                        for split_op in op.split()[
                            ::-1
                        ]:  # AB|f> = A(B|f>) i.e. B comes first
                            # if (
                            #     next(iter(op.op_dict)) == " "
                            # ):  # Identity * |state> = |state>
                            #     state, new_coeff = state, 1
                            # else:
                            state, new_coeff = split_op._operate_on_state(state)
                            split_coeff *= (
                                new_coeff  # Update coeff for every op in product
                            )
                        output_state_dict[next(iter(state.state_dict))] = (
                            split_coeff * op_coeff * state_coeff
                        ) + output_state_dict.get(next(iter(state.state_dict)), 0)
            return Fock(state_dict=output_state_dict)._cleanup()

    def __add__(self, other: "Fock") -> "Fock":
        # (TODO: could uses sets to find common and different keys and loop only over unique terms)
        # loop over smaller dict
        if len(self.state_dict) < len(other.state_dict):
            new_dict = deepcopy(other.state_dict)
            for state, coeff in self.state_dict.items():
                new_dict[state] = coeff + new_dict.get(state, 0)
        else:
            new_dict = deepcopy(self.state_dict)
            for state, coeff in other.state_dict.items():
                new_dict[state] = coeff + new_dict.get(state, 0)
        out_state = Fock(state_dict=new_dict)._cleanup()
        if len(out_state.state_dict) == 0:
            return 0
        else:
            return out_state

    def _cleanup(self, zero_threshold=1e-15) -> "Fock":
        """
        remove terms below threshold
        """
        if len(self.state_dict) == 0:
            return self
        else:
            # keys, coeffs = zip(*self.state_dict.items())
            # mask = np.where(abs(np.array(coeffs)) > zero_threshold)[0]
            # new_state_dict = dict(zip(np.take(keys, mask), np.take(coeffs, mask)))
            # return Fock(state_dict=new_state_dict)

            ## SLOWER than above, but nested tuples causes problems in numpy
            ## TODO: speed this up in numba // use numpy mask but do NOT use to mask keys... instead loop over mask as below
            new_state_dict = dict()
            for op, coeff in self.state_dict.items():
                if abs(coeff) > zero_threshold:
                    new_state_dict[op] = coeff

            return Fock(state_dict=new_state_dict)

    def dagger(self):
        return ConjugateFock(state_dict=self.state_dict)

    def __sub__(self, other: "Fock") -> "Fock":
        coeffs = list(other.state_dict.values())
        neg_other = Fock(
            state_dict=dict(zip(other.state_dict.keys(), -1 * np.array(coeffs)))
        )
        return self + neg_other


class ConjugateFock:
    """
    Defines a Conjugate Fock (bra) state of the form ⟨f, \bar{f}, b|

    Parameters:
    ferm_occupancy, antiferm_occupancy, bos_occupancy: List
    e.g. a hadron with a fermion in mode 2, an antifermion in mode 1 and 2 bosons
    in mode 1 is
    ferm_occupancy = [2]
    antiferm_occupancy = [1]
    bos_occupancy = [(1, 2)]

    """

    def __init__(
        self,
        f_occ: List = None,
        af_occ: List = None,
        b_occ: List = None,
        state_dict: Dict[Tuple[Tuple, Tuple, Tuple], complex] = None,
        coeff: complex = 1.0,
        perform_cleanup=True,
    ):
        if f_occ is not None and af_occ is not None and b_occ is not None:
            self.state_dict = {
                (
                    tuple(f_occ),
                    tuple(af_occ),
                    tuple([(n, m) for (n, m) in b_occ if m != 0]),
                ): coeff
            }

        elif state_dict is not None:
            self.state_dict = state_dict

    def __str__(self):
        if len(self.state_dict) == 0:
            return "0"
        else:
            output_str = ""
            for state, coeff in self.state_dict.items():
                output_str += f"{coeff} * ⟨{state}| +"
                output_str += "\n"
            return output_str[:-3]

    def __repr__(self) -> str:
        return self.__str__()

    def display(self):
        display(Latex("$" + self.__str__() + "$"))

    def __rmul__(self, other):
        if isinstance(other, (float, int, complex)):
            coeffs = np.array(list(self.state_dict.values()))
            return ConjugateFock(
                state_dict=dict(zip(self.state_dict.keys(), other * coeffs))
            )

    def __add__(self, other: "ConjugateFock") -> "ConjugateFock":
        # (TODO: could uses sets to find common and different keys and loop only over unique terms)
        # loop over smaller dict
        if len(self.state_dict) < len(other.state_dict):
            new_dict = deepcopy(other.state_dict)
            for state, coeff in self.state_dict.items():
                new_dict[state] = coeff + new_dict.get(state, 0)
        else:
            new_dict = deepcopy(self.state_dict)
            for state, coeff in other.state_dict.items():
                new_dict[state] = coeff + new_dict.get(state, 0)

        return ConjugateFock(state_dict=new_dict, perform_cleanup=True)

    def dagger(self):
        return Fock(state_dict=self.state_dict)

    def inner_product(self, other: "Fock") -> complex:
        if isinstance(other, Fock):
            inner_product = 0
            for state1, coeff1 in self.state_dict.items():
                for state2, coeff2 in other.state_dict.items():
                    if state1 == state2:
                        inner_product += 1.0 * coeff1 * coeff2
            return inner_product

    def __mul__(self, other):
        if isinstance(other, Fock):
            return self.inner_product(other)
        elif isinstance(other, ParticleOperator):
            out = other.dagger() * self.dagger()
            if isinstance(out, (int, float, complex)):
                return 0
            else:
                return out.dagger()  # (<f|A)^\dagger = (A^\dagger * |f>)^\dagger

    def __sub__(self, other: "ConjugateFock") -> "ConjugateFock":
        coeffs = list(other.state_dict.values())
        neg_other = ConjugateFock(
            state_dict=dict(zip(other.state_dict.keys(), -1 * np.array(coeffs)))
        )
        return self + neg_other


class ParticleOperator:

    def __init__(self, op_dict: Union[Dict[str, complex], str] = dict()):

        if isinstance(op_dict, str):
            self.op_dict = {op_dict: 1}
        elif isinstance(op_dict, dict):
            self.op_dict = op_dict
        else:
            raise ValueError("input must be dictionary or op string")

    def __add__(self, other: "ParticleOperator") -> "ParticleOperator":

        # (TODO: could uses sets to find common and different keys and loop only over unique terms)
        # loop over smaller dict
        if len(self.op_dict) < len(other.op_dict):
            new_dict = deepcopy(other.op_dict)
            for op_str, coeff in self.op_dict.items():
                new_dict[op_str] = coeff + new_dict.get(op_str, 0)
        else:
            new_dict = deepcopy(self.op_dict)
            for op_str, coeff in other.op_dict.items():
                new_dict[op_str] = coeff + new_dict.get(op_str, 0)

        return ParticleOperator(new_dict)._cleanup()

    def __str__(self) -> str:
        if len(self.op_dict) == 0:
            return "0"
        else:
            output_str = ""
            for op, coeff in self.op_dict.items():
                output_str += f"{coeff} * {op}"
                output_str += "\n"
            return output_str

    def _cleanup(self, zero_threshold=1e-15) -> "ParticleOperator":
        """
        remove terms below threshold
        """
        if len(self.op_dict) == 0:
            return self
        else:
            keys, coeffs = zip(*self.op_dict.items())
            mask = np.where(abs(np.array(coeffs)) > zero_threshold)[0]
            new_op_dict = dict(zip(np.take(keys, mask), np.take(coeffs, mask)))
            # new_op_dict = dict()
            # for idx in mask:
            #     new_op_dict[keys[idx]] = coeffs[idx]
            return ParticleOperator(new_op_dict)

    def __repr__(self) -> str:
        return self.__str__()

    def latex_string(self) -> Latex:
        pattern = r"(\d+)"
        replacement = r"_{\1}"
        latex_str = ""
        for s, coeff in self.op_dict.items():
            new_s = s.replace("^", "^{\dagger}")
            latex_str += f"{coeff} {re.sub(pattern, replacement, new_s)} +"

        return Latex("$" + latex_str[:-1] + "$")

    def display(self):
        display(self.latex_string())

    def split(self) -> List:
        split_list = []
        if len(self.op_dict) == 1:
            for oper in list(self.op_dict.keys())[0].split(" "):
                if oper[0] == "a":  # boson
                    split_list.append(BosonOperator(oper[1:]))
                elif oper[0] == "b":  # fermion
                    split_list.append(FermionOperator(oper[1:]))
                elif oper[0] == "d":  # antifermion
                    split_list.append(AntifermionOperator(oper[1:]))
            return split_list
        else:
            return NotImplemented

    def to_list(self) -> List:
        particle_op_list = []
        for oper, coeff in self.op_dict.items():
            particle_op_list.append(ParticleOperator({oper: coeff}))
        return particle_op_list

    def dagger(self) -> "ParticleOperator":
        dagger_dict = {}

        for i, coeff_i in self.op_dict.items():
            dagger_str = ""
            if i == " ":  # I^dagger = I
                dagger_dict[" "] = coeff_i.conjugate()
            else:
                for oper in i.split(" ")[::-1]:
                    if oper[-1] == "^":
                        dagger_str += oper[:-1]
                    else:
                        dagger_str += oper + "^"
                    dagger_str += " "
                dagger_dict[dagger_str[:-1]] = coeff_i.conjugate()

        return ParticleOperator(dagger_dict)

    def __rmul__(self, other):
        coeffs = np.array(list(self.op_dict.values()))
        return ParticleOperator(dict(zip(self.op_dict.keys(), other * coeffs)))

    def __mul__(self, other):
        if isinstance(other, ParticleOperator):
            product_dict = {}
            for op1, coeffs1 in list(self.op_dict.items()):
                for op2, coeffs2 in list(other.op_dict.items()):
                    if op1 == " " and op2 == " ":
                        product_dict[" "] = coeffs1 * coeffs2
                    else:
                        # Add .strip() to remove trailing spaces when multipying with identity (treated as ' ')
                        product_dict[(op1 + " " + op2).strip()] = coeffs1 * coeffs2
            return ParticleOperator(product_dict)
        return NotImplemented

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

    @property
    def coeff(self):
        assert (
            len(self) == 1
        ), "coeff property exists for 1 element in op_dict only. Use .coeffs property"
        return next(iter(self.op_dict.values()))

    @property
    def coeffs(self):
        return list(self.op_dict.values())

    def normal_order(self) -> "ParticleOperator":

        # Returns a new ParticleOperator object with a normal ordered hash table
        # normal ordering: b^dagger before b; d^dagger before d; a^dagger before a
        # b2 b1^ a0 b3 -> b1^ b2 b3 a0
        # return ParticleOperator(normal_ordered_dict)
        ordered_op = ParticleOperator({})
        po_list = self.to_list()

        # for each PO in POS, convert it to normal order
        for particle_op in po_list:
            coeff_op = list(particle_op.op_dict.values())[0]
            normal_ordered_op = particle_op._normal_order(coeff_op)
            ordered_op += normal_ordered_op
        return ordered_op

    # inner func only operates on one particle operator instance
    # i.e. ParticleOperator("a0 b0 a2^ b0^ b0^ b1 d3 a2^ d3^")
    def _normal_order(self, coeff) -> "ParticleOperator":
        # prevent normal ordering identity/empty op
        if list(self.op_dict.keys())[0].strip() == "":
            return self

        # parse the op_str into bs, ds, and as and normal ordering them separately
        op_list = self.split()
        fermion_list = []
        antifermion_list = []
        boson_list = []

        swap_bd = 0
        for op in op_list:
            if isinstance(op, FermionOperator):
                swap_bd += len(antifermion_list)
                fermion_list.append(op)
            elif isinstance(op, AntifermionOperator):
                antifermion_list.append(op)
            elif isinstance(op, BosonOperator):
                boson_list.append(op)
            else:
                raise Exception("Unknown particle type appear in normal order")

        # insertion sort the ops
        # note that extra terms will be added for nontrivial swaps
        # normal_{x}s is a particle operator that contains the key with particle op only in type x
        normal_bs = self.insertion_sort(
            fermion_list, 0 if len(fermion_list) == 0 else 1
        )
        normal_ds = self.insertion_sort(
            antifermion_list, 0 if len(antifermion_list) == 0 else 1
        )
        normal_as = self.insertion_sort(boson_list, 0 if len(boson_list) == 0 else 1)

        # sign change between swapping bs and ds
        if (swap_bd % 2) == 0:
            sign_swap_bd = 1
        else:
            sign_swap_bd = -1

        # create the normal ordered particle operator and restore the coeff
        if normal_bs.op_dict and normal_ds.op_dict and normal_as.op_dict:
            ordered_op = sign_swap_bd * normal_bs * normal_ds * normal_as
        elif normal_bs.op_dict and normal_ds.op_dict:
            ordered_op = sign_swap_bd * normal_bs * normal_ds
        elif normal_bs.op_dict and normal_as.op_dict:
            ordered_op = normal_bs * normal_as
        elif normal_ds.op_dict and normal_as.op_dict:
            ordered_op = normal_ds * normal_as
        elif normal_bs.op_dict:
            ordered_op = normal_bs
        elif normal_ds.op_dict:
            ordered_op = normal_ds
        elif normal_as.op_dict:
            ordered_op = normal_as

        for key in ordered_op.op_dict.keys():
            ordered_op.op_dict[key] *= coeff

        return ordered_op

    # ops can get from op.split()
    def is_normal_ordered(self, ops):
        found_non_creation = False
        for op in ops:
            if op.creation:
                if found_non_creation:
                    return False
            else:
                found_non_creation = True
        return True

    def insertion_sort(self, ops, coeff) -> "ParticleOperator":
        # base case(s)
        if not ops or coeff == 0:
            return ParticleOperator({})

        result_op = ParticleOperator({})

        if self.is_normal_ordered(ops):
            # return the op with normal ordered key
            if len(ops) != 0:
                result_str = ""
                for i in range(len(ops)):
                    result_str += list(ops[i].op_dict.keys())[0] + " "
                result_op = ParticleOperator(result_str[:-1])
                result_op.op_dict[result_str[:-1]] = coeff
            return result_op

        # recursive case
        else:
            if isinstance(ops[0], BosonOperator):
                type_coeff = 1
                is_boson = True
            else:
                type_coeff = -1
                is_boson = False

            for i in range(1, len(ops)):
                right = ops[i]
                j = i - 1

                while (
                    j >= 0
                    and not ops[j].creation
                    and not (not ops[j].creation and not right.creation)
                ):
                    # case for non-trivial swap --> introduce extra terms
                    # ops[j] is the left annihilation operator
                    if ops[j].mode == right.mode:
                        # for bosons: a_i a_i^ = 1 + a_i^ a_i
                        # for (anti)fermions: b_i b_i^ = 1 - b_i^ b_i (same for 'd's)

                        # declare the identity operator
                        identity = ParticleOperator(" ")
                        new_op_str = (
                            list(right.op_dict.keys())[0]
                            + " "
                            + list(ops[j].op_dict.keys())[0]
                        )
                        new_op = coeff * (
                            identity + type_coeff * ParticleOperator(new_op_str)
                        )

                        left_op = self.combine_op(ops, 0, j)
                        right_op = self.combine_op(ops, j + 2, len(ops))

                        tree_split = left_op * new_op * right_op

                        for key_op, val_op in tree_split.op_dict.items():
                            if key_op == "":
                                identity = ParticleOperator(" ")
                                identity.op_dict[" "] = coeff
                                result_op += identity
                            else:
                                recur_ops = ParticleOperator(key_op).split()
                                result_op += self.insertion_sort(recur_ops, val_op)

                        return result_op

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
                result_str = ""
                for i in range(len(ops)):
                    result_str += list(ops[i].op_dict.keys())[0] + " "
                result_op = ParticleOperator(result_str[:-1])
                result_op.op_dict[result_str[:-1]] = coeff
            return result_op

    # helper func to get the particle operator from index i to j as a single instance
    def combine_op(self, ops, i, j):
        result_str = ""
        for op in range(len(ops)):
            if op >= i and op < j:
                result_str += list(ops[op].op_dict.keys())[0] + " "
        return ParticleOperator(result_str[:-1])

    def max_mode(self):
        all_ops_as_str = " ".join(list(self.op_dict.keys()))
        return max(list(map(lambda x: int(x), re.findall(r"\d+", all_ops_as_str))))

    def commutator(self, other: "ParticleOperator") -> "ParticleOperator":
        # [A, B] = AB - BA
        return (self * other).normal_order() - (other * self).normal_order()

    def anticommutator(self, other: "ParticleOperator") -> "ParticleOperator":
        # {A, B} = AB + BA
        return (self * other).normal_order() + (other * self).normal_order()


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
            # Get parity
            coeff = (-1) ** len(f_occ[: f_occ.index(self.mode)])
        elif not self.creation and self.mode in f_occ:
            # Get parity
            coeff = (-1) ** len(f_occ[: f_occ.index(self.mode)])
            f_occ.remove(self.mode)
        else:
            return Fock([], [], []), 0

        f_occ = sorted(f_occ)

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
            coeff = (-1) ** len(af_occ[: af_occ.index(self.mode)])
        elif not self.creation and self.mode in af_occ:
            # Get parity
            coeff = (-1) ** len(af_occ[: af_occ.index(self.mode)])
            af_occ.remove(self.mode)
        else:
            return Fock([], [], []), 0

        af_occ = sorted(af_occ)

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


class NumberOperator(ParticleOperator):
    def __init__(self, particle_type, mode):
        self.particle_type = particle_type
        self.mode = mode
        super().__init__(
            self.particle_type
            + str(self.mode)
            + "^ "
            + self.particle_type
            + str(self.mode),
        )

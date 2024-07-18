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
        output_str = ""
        for state, coeff in self.state_dict.items():
            output_str += f"{coeff} * |{state}⟩ +"
            output_str += "\n"
        return output_str[:-3]

    def __repr__(self) -> str:
        return self.__str__()

    def display(self):
        display(Latex("$" + self.__str__() + "$"))

    def __rmul__(self, other):
        if isinstance(other, (float, int, complex)):
            coeffs = np.array(list(self.state_dict.values()))
            return Fock(state_dict=dict(zip(self.state_dict.keys(), other * coeffs)))
        elif isinstance(other, Fock):
            raise Exception("Cannot multiply a ket to the left of a ket")

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

        return Fock(state_dict=new_dict, perform_cleanup=True)

    def dagger(self):
        return ConjugateFock(state_dict=self.state_dict)


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
        Fock(state_dict=self.state_dict)

    def inner_product(self, other: "Fock") -> complex:
        if isinstance(other, Fock):
            inner_product = 0
            for state1, coeff1 in self.state_dict.items():
                for state2, coeff2 in other.state_dict.items():
                    if state1 == state2:
                        inner_product += 1.0 * coeff1 * coeff2
            return inner_product

    def __mul__(self, other: "Fock") -> "Fock":
        return self.inner_product(other)


class ParticleOperator:

    def __init__(
        self, op_dict: Union[Dict[str, complex], str] = dict(), perform_cleanup=True
    ):

        if isinstance(op_dict, str):
            self.op_dict = {op_dict: 1}
        elif isinstance(op_dict, dict):
            self.op_dict = op_dict
        else:
            raise ValueError("input must be dictionary or op string")

        if perform_cleanup:
            self.cleanup()

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

        return ParticleOperator(new_dict, perform_cleanup=True)

    def __str__(self) -> str:
        output_str = ""
        for op, coeff in self.op_dict.items():
            output_str += f"{coeff} * {op}"
            output_str += "\n"
        return output_str

    def cleanup(self, zero_threshold=1e-15) -> None:
        """
        remove terms below threshold
        """
        keys, coeffs = zip(*self.state_dict.items())
        mask = np.where(abs(np.array(coeffs)) > 1e-15)[0]
        self.state_dict = dict(zip(np.take(keys, mask), np.take(coeffs, mask)))
        # self.state_dict = dict()
        # for idx in mask:
        #     self.state_dict[keys[idx]] = coeffs[idx]
        return None

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
            for oper in i.split(" ")[::-1]:
                if oper[-1] == "^":
                    dagger_str += oper[:-1]
                else:
                    dagger_str += oper + "^"
                dagger_str += " "
            dagger_dict[dagger_str[:-1]] = coeff_i.conjugate()

        return ParticleOperator(dagger_dict)

    def cleanup(self, zero_threshold=1e-15) -> None:
        """
        remove terms below threshold
        """
        keys, coeffs = zip(*self.op_dict.items())
        mask = np.where(abs(np.array(coeffs)) > 1e-15)[0]
        self.op_dict = dict(zip(np.take(keys, mask), np.take(coeffs, mask)))
        # self.op_dict = dict()
        # for idx in mask:
        #     self.op_dict[keys[idx]] = coeffs[idx]
        return None

    def __rmul__(self, other):
        coeffs = np.array(list(self.op_dict.values()))
        return ParticleOperator(dict(zip(self.op_dict.keys(), other * coeffs)))

    def __mul__(self, other):
        if isinstance(other, ParticleOperator):
            product_dict = {}
            for op1, coeffs1 in list(self.op_dict.items()):
                for op2, coeffs2 in list(other.op_dict.items()):
                    product_dict[op1 + " " + op2] = coeffs1 * coeffs2
            return ParticleOperator(product_dict)
        if isinstance(other, Fock):
            # TODO: Implement
            pass

    def __pow__(self, other) -> "ParticleOperator":
        if len(self.op_dict) == 1:
            return ParticleOperator(
                ((str(list(self.op_dict.keys())[0]) + " ") * other)[:-1]
            )
        else:
            return NotImplemented

    def __sub__(self, other):
        coeffs = list(other.op_dict.values())
        neg_other = ParticleOperator(
            dict(zip(other.op_dict.keys(), -1 * np.array(coeffs)))
        )
        return self + neg_other

    def impose_parity_jw(self, state, mode):

        index = state.index(mode)
        parity_str = state[:index]
        coeff = (-1) ** len(parity_str)
        return coeff

    def operate_on_state(self, other):
        if isinstance(other, (int, float)):
            return other
        else:
            coeff = self.coeff * other.coeff
            updated_ferm_state = other.f_occ[:]
            updated_antiferm_state = other.af_occ[:]
            updated_bos_state = other.b_occ[:]

            op = self.input_string
            if op[-1] == "^":
                if op[0] == "b":
                    if int(op[1]) not in other.f_occ:
                        updated_ferm_state.append(int(op[1]))
                        coeff *= self.impose_parity_jw(
                            sorted(updated_ferm_state), int(op[1])
                        )
                    else:
                        coeff = 0
                elif op[0] == "d":
                    if int(op[1]) not in other.af_occ:
                        updated_antiferm_state.append(int(op[1]))
                        coeff *= self.impose_parity_jw(
                            sorted(updated_antiferm_state), int(op[1])
                        )
                    else:
                        coeff = 0
                elif op[0] == "a":
                    state_modes, state_occupancies = [i[0] for i in other.b_occ], [
                        i[1] for i in other.b_occ
                    ]

                    if int(op[1]) in state_modes:
                        index = state_modes.index(int(op[1]))
                        if state_occupancies[index] >= 0:
                            state_occupancies[index] += 1
                            coeff *= np.sqrt(state_occupancies[index])

                    else:
                        state_modes.append(int(op[1]))
                        state_occupancies.append(1)
                    # zip up modes and occupancies into an updated list
                    updated_bos_state = list(zip(state_modes, state_occupancies))
                    sorted_updated_bos_state = sorted(
                        updated_bos_state, key=lambda x: x[0]
                    )
            else:
                if op[0] == "b":
                    if int(op[1]) in other.f_occ:
                        coeff *= self.impose_parity_jw(updated_ferm_state, int(op[1]))
                        updated_ferm_state.remove(int(op[1]))

                    else:
                        coeff = 0
                elif op[0] == "d":
                    if int(op[1]) in other.af_occ:
                        coeff *= self.impose_parity_jw(
                            updated_antiferm_state, int(op[1])
                        )
                        updated_antiferm_state.remove(int(op[1]))
                    else:
                        coeff = 0
                elif op[0] == "a":
                    state_modes, state_occupancies = [i[0] for i in other.b_occ], [
                        i[1] for i in other.b_occ
                    ]
                    if int(op[1]) in state_modes:
                        index = state_modes.index(int(op[1]))
                        if state_occupancies[index] >= 1:
                            state_occupancies[index] -= 1
                            coeff *= np.sqrt(state_occupancies[index] + 1)
                        else:
                            coeff = 0
                    else:
                        coeff = 0
                    # zip up modes and occupancies into an updated list
                    updated_bos_state = list(zip(state_modes, state_occupancies))
                    sorted_updated_bos_state = sorted(
                        updated_bos_state, key=lambda x: x[0]
                    )

            if "a" in self.particle_type:
                return coeff * Fock(
                    sorted(updated_ferm_state),
                    sorted(updated_antiferm_state),
                    sorted_updated_bos_state,
                )
            else:
                return coeff * Fock(
                    sorted(updated_ferm_state),
                    sorted(updated_antiferm_state),
                    updated_bos_state,
                )


class FermionOperator(ParticleOperator):

    def __init__(self, mode, coeff: complex = 1.0):
        self.mode = mode
        self.particle_type = "fermion"
        super().__init__("b" + mode, coeff)

    def __str__(self):
        return super().__str__()

    def __mul__(self, other):
        return super().__mul__(other)

    def __add__(self, other):
        return super().__add__(other)


class AntifermionOperator(ParticleOperator):
    def __init__(self, mode, coeff: complex = 1.0):
        self.mode = mode
        self.particle_type = "antifermion"
        super().__init__("d" + mode, coeff)

    def __str__(self):
        return super().__str__()

    def __mul__(self, other):
        return super().__mul__(other)

    def __add__(self, other):
        return super().__add__(other)


class BosonOperator(ParticleOperator):
    def __init__(self, mode, coeff: complex = 1.0):
        self.mode = mode
        self.particle_type = "boson"
        super().__init__("a" + mode, coeff)

    def __str__(self):
        return super().__str__()

    def __mul__(self, other):
        return super().__mul__(other)

    def __add__(self, other):
        return super().__add__(other)

import numpy as np
from sympy import *
from typing import List, Union, Optional, Dict
from IPython.display import display, Latex
from collections import defaultdict
import re
from copy import deepcopy


class Fock:
    """
    Defines a Fock state of the form |f, \bar{f}, b⟩

    Parameters:
    ferm_occupancy, antiferm_occupancy, bos_occupancy: List
    e.g. a hadron with a fermion in mode 2, an antifermion in mode 1 and 2 bosons
    in mode 1 is
    ferm_occupancy = [2]
    antiferm_occupancy = [1]
    bos_occupancy = [(1, 2)]

    """

    def __init__(self, f_occ, af_occ, b_occ, coeff: complex = 1.0):
        self.f_occ = f_occ
        self.af_occ = af_occ
        self.b_occ = [(n, m) for (n, m) in b_occ if m != 0]
        self.coeff = coeff

    def __str__(self):
        return (
            str(self.coeff)
            + " * |"
            + ",".join([str(i) for i in self.f_occ][::-1])
            + "; "
            + ",".join([str(i) for i in self.af_occ][::-1])
            + "; "
            + ",".join([str(i) for i in self.b_occ][::-1])
            + "⟩"
        )

    def display(self):
        display(Latex("$" + self.__str__() + "$"))

    def __eq__(self, other):
        if isinstance(other, Fock):
            return (
                self.f_occ == other.f_occ
                and self.af_occ == other.af_occ
                and self.b_occ == other.b_occ
            )

    def __rmul__(self, other):
        if isinstance(other, (float, int, complex)):
            if other == 0:
                return 0
            else:
                coeff = self.coeff * other
                return Fock(self.f_occ, self.af_occ, self.b_occ, coeff)
        elif isinstance(other, ConjugateFock):
            raise NotImplemented
        elif isinstance(other, Fock):
            raise Exception("Cannot multiply a ket to the left of a ket")

    def __mul__(self, other):
        # raise NotImplemented
        return None

    def __add__(self, other):
        if isinstance(other, Fock):
            return FockSum([self, other]).cleanup()
        elif isinstance(other, FockSum):
            return FockSum([self] + other.states_list).cleanup()

    def dagger(self):
        return ConjugateFock.from_state(self)


class ConjugateFock:

    def __init__(self, f_occ, af_occ, b_occ, coeff: complex = 1.0):
        self.f_occ = f_occ
        self.af_occ = af_occ
        self.b_occ = b_occ
        self.coeff = coeff

    def __str__(self):
        return (
            str(self.coeff)
            + " * "
            + "⟨"
            + ",".join([str(i) for i in self.f_occ][::-1])
            + "; "
            + ",".join([str(i) for i in self.af_occ][::-1])
            + "; "
            + ",".join([str(i) for i in self.b_occ][::-1])
            + "|"
        )

    @classmethod
    def from_state(cls, state: Fock):
        f_occ = state.f_occ
        af_occ = state.af_occ
        b_occ = state.b_occ
        coeff = state.coeff.conjugate()
        return cls(f_occ, af_occ, b_occ, coeff)

    def display(self):
        display(Latex("$" + self.__str__() + "$"))

    def inner_product(self, other):
        if (
            self.f_occ == other.f_occ
            and self.af_occ == other.af_occ
            and self.b_occ == other.b_occ
        ):
            return 1.0 * self.coeff * other.coeff
        else:
            return 0

    def dagger(self):
        return Fock(self.f_occ, self.af_occ, self.b_occ, self.coeff.conjugate())

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return ConjugateFock(self.f_occ, self.af_occ, self.b_occ, coeff=other)

    def __add__(self, other):
        if isinstance(other, ConjugateFock):
            return ConjugateFockSum([self, other]).cleanup()
        elif isinstance(other, ConjugateFockSum):
            return ConjugateFockSum([self] + other.states_list).cleanup()

    def __eq__(self, other):
        if isinstance(other, ConjugateFock):
            return (
                self.f_occ == other.f_occ
                and self.af_occ == other.af_occ
                and self.b_occ == other.b_occ
            )

    def __mul__(self, other):
        if isinstance(other, Fock):
            return self.inner_product(other)
        if isinstance(other, FockSum):
            output_value = 0
            for state in other.states_list:
                output_value += self * state
            return output_value
        elif isinstance(other, ParticleOperator):
            # <f|A = (A^dagger |f>)^dagger
            out_state = other.dagger() * self.dagger()
            if isinstance(out_state, (int, float)):
                return out_state
            else:
                return out_state.dagger()
        elif isinstance(other, ParticleOperatorSum):
            # <f|(A + B) = ((A^dagger + B^dagger)|f>)^dagger
            out_state = other.dagger() * self.dagger()
            if isinstance(out_state, (int, float)):
                return out_state
            else:
                return (out_state).dagger()


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

    def split(self):
        pass

    def dagger(self):
        pass

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
        pass

    def __pow__(self, other):
        pass

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

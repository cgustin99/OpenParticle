import numpy as np
from sympy import *
from typing import List
from IPython.display import display, Latex
from collections import defaultdict


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

    def __init__(self, f_occ, af_occ, b_occ, coeff: float = 1.0):
        self.f_occ = f_occ
        self.af_occ = af_occ
        self.b_occ = b_occ
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
        if isinstance(other, (float, int)):
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
        raise NotImplemented

    def __add__(self, other):
        if isinstance(other, Fock):
            return FockSum([self, other]).cleanup()

    def dagger(self):
        return ConjugateFock.from_state(self)


class ConjugateFock:

    def __init__(self, f_occ, af_occ, b_occ, coeff: float = 1.0):
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
        coeff = state.coeff
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
        return Fock(self.f_occ, self.af_occ, self.b_occ, self.coeff)

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return ConjugateFock(self.f_occ, self.af_occ, self.b_occ, coeff=other)

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
                return (
                    other.coeff * self.coeff * (other.dagger() * self.dagger()).dagger()
                )
        elif isinstance(other, ParticleOperatorSum):
            # <f|(A + B) = ((A^dagger + B^dagger)|f>)^dagger
            out_state = other.dagger() * self.dagger()
            if isinstance(out_state, (int, float)):
                return out_state
            else:
                return (out_state).dagger()


class FockSum:
    def __init__(self, states_list: List[Fock]):
        self.states_list = states_list

    def normalize(self):
        # TODO
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, FockSum):
            return [i for i in self.states_list] == other.states_list

    def cleanup(self):
        output_list_of_states = []
        coeff_counter = 0
        for i in range(len(self.states_list)):
            if self.states_list[i] not in output_list_of_states:
                coeff_counter += self.states_list[
                    i
                ].coeff  # add coeff of the first operator
                for j in range(i + 1, len(self.states_list)):
                    if (
                        self.states_list[i].f_occ == self.states_list[j].f_occ
                        and self.states_list[i].af_occ == self.states_list[j].af_occ
                        and self.states_list[i].b_occ == self.states_list[j].b_occ
                    ):
                        coeff_counter += self.states_list[
                            j
                        ].coeff  # add coeffs of other same operators in the list

                output_list_of_states.append(
                    Fock(
                        self.states_list[i].f_occ,
                        self.states_list[i].af_occ,
                        self.states_list[i].b_occ,
                        coeff_counter,
                    )
                )

                coeff_counter = 0
        if len(output_list_of_states) == 1:
            return output_list_of_states[0]
        else:
            return FockSum(output_list_of_states)

    def display(self):
        return display(Latex("$" + self.__str__() + "$"))

    def __str__(self):
        states_str = ""
        for index, state in enumerate(self.states_list):
            if index != len(self.states_list) - 1:
                states_str += state.__str__() + " + "
            else:
                states_str += state.__str__()

        return states_str

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            # const. * (|a> + |b>)
            out_states = []
            for state in self.states_list:
                out_states.append(other * state)
            return FockSum(out_states)

    def dagger(self):
        out_state = []

        for op in self.states_list:
            out_state.append(op.dagger())
        return ConjugateFockSum(out_state)


class ConjugateFockSum:

    def __init__(self, states_list: List[ConjugateFock]):
        self.states_list = states_list

    def display(self):
        return display(Latex("$" + self.__str__() + "$"))

    def __str__(self):
        states_str = ""
        for index, state in enumerate(self.states_list):
            if index != len(self.states_list) - 1:
                states_str += state.__str__() + " + "
            else:
                states_str += state.__str__()

        return states_str

    def dagger(self):
        out_state = []

        for op in self.states_list:
            out_state.append(op.dagger())
        return FockSum(out_state)

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            out_states = []
            for state in self.states_list:
                out_states.append(other * state)
            return ConjugateFockSum(out_states)

    def __mul__(self, other):
        if isinstance(other, Fock):
            output_value = 0
            for conj_state in self.states_list:
                output_value += conj_state * other
            return output_value
        elif isinstance(other, FockSum):
            output_value = 0
            for conj_state in self.states_list:
                for state in other.states_list:
                    output_value += conj_state * state
            return output_value
        elif isinstance(other, ParticleOperator):
            output_conj_states = []
            for conj_state in self.states_list:
                output_conj_states.append(conj_state * other)
            return ConjugateFockSum(output_conj_states)
        elif isinstance(other, ParticleOperatorSum):
            output_conj_states = []
            for conj_state in self.states_list:
                for op in other.operator_list:
                    output_conj_states.append(conj_state * op)
            return ConjugateFockSum(output_conj_states)


class ParticleOperator:

    def __init__(self, input_string, coeff=1.0):
        # Initializes particle operator in the form of b_n d_n a_n
        # Parameters:
        # input_string: e.g. 'b2^ a0'

        self.input_string = input_string
        self.coeff = coeff

        particle_type = ""
        modes = []
        ca_string = ""

        for op in self.input_string.split(" "):
            type = op[0]
            orbital = op[1]
            particle_type += type
            modes.append(int(orbital))
            if op[-1] != "^":
                ca_string += "a"
            else:
                ca_string += "c"

        self.particle_type = particle_type
        self.modes = modes
        self.ca_string = ca_string

        fermion_modes = []
        antifermion_modes = []
        boson_modes = []

        op_string = ""
        for index, particle in enumerate(self.particle_type):

            if particle == "b":
                fermion_modes.append(self.modes[index])
                if self.ca_string[index] == "c":
                    op_string += "b^†_" + str(self.modes[index])
                else:
                    op_string += "b_" + str(self.modes[index])

            elif particle == "d":
                antifermion_modes.append(self.modes[index])
                if self.ca_string[index] == "c":
                    op_string += "d^†_" + str(self.modes[index])
                else:
                    op_string += "d_" + str(self.modes[index])

            elif particle == "a":
                boson_modes.append(self.modes[index])
                if self.ca_string[index] == "c":
                    op_string += "a^†_" + str(self.modes[index])
                else:
                    op_string += "a_" + str(self.modes[index])

        self.op_string = str(self.coeff) + "*" + op_string
        self.fermion_modes = fermion_modes
        self.antifermion_modes = antifermion_modes
        self.boson_modes = boson_modes

    def __str__(self):
        return self.op_string

    def __eq__(self):
        if isinstance(other, ParticleOperator):
            return self.input_string == other.input_string

    @staticmethod
    def op_list_dagger(list):
        dag_list = []
        for op in list[::-1]:
            if op[-1] == "^":
                dag_list.append(op[:-1])
            else:
                dag_list.append(op + "^")
        return dag_list

    def dagger(self):
        elements = self.input_string.split(" ")

        grouped_elements = defaultdict(list)
        for element in elements:
            first_letter = element[0]
            grouped_elements[first_letter].append(element)

        grouped_lists = list(grouped_elements.values())

        list_b = grouped_elements["b"]
        list_d = grouped_elements["d"]
        list_a = grouped_elements["a"]

        dag_op_string = " ".join(
            self.op_list_dagger(list_b)
            + self.op_list_dagger(list_d)
            + self.op_list_dagger(list_a)
        )

        return ParticleOperator(dag_op_string, self.coeff)

    def display(self):
        display(Latex("$" + self.op_string + "$"))

    def __add__(self, other):
        if isinstance(other, ParticleOperator):
            if self.input_string == other.input_string:
                # A + A = 2A
                return ParticleOperator(self.input_string, self.coeff + other.coeff)
            else:
                # A + B = A + B (ParticleOperatorSum Object)
                return ParticleOperatorSum([self, other])
        elif isinstance(other, ParticleOperatorSum):
            # A + (B + C)
            output = [self] + other.operator_list
            return ParticleOperatorSum(output).cleanup()

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            self.coeff *= other
            return ParticleOperator(self.input_string, self.coeff)

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

    def __mul__(self, other):
        if isinstance(other, ParticleOperator):
            updated_coeff = self.coeff * other.coeff
            return ParticleOperator(
                self.input_string + " " + other.input_string, updated_coeff
            )

        elif isinstance(other, Fock):
            po_coeff = self.coeff
            for op in self.input_string.split(" ")[::-1]:
                other = ParticleOperator(op).operate_on_state(other)

            return po_coeff * other

        elif isinstance(other, FockSum):
            updated_states = []
            for state in other.states_list:
                state_prime = self.operate_on_state(state)
                if isinstance(state_prime, Fock):
                    updated_states.append(state_prime)
            return FockSum(updated_states)


class ParticleOperatorSum:
    # Sum of ParticleOperator instances
    def __init__(self, operator_list: List[ParticleOperator]):
        self.operator_list = operator_list

    def __str__(self):
        op_string = ""
        for index, op in enumerate(self.operator_list):
            if index != len(self.operator_list) - 1:
                op_string += op.__str__() + " + "
            else:
                op_string += op.__str__()
        return op_string

    def display(self):
        return display(Latex("$" + self.__str__() + "$"))

    def dagger(self):
        out_ops = []
        for op in self.operator_list:
            out_ops.append(op.dagger())
        return ParticleOperatorSum(out_ops)

    def get_modes(self):
        modes_list = []

        for op in self.operator_list:
            modes_list.append(tuple(op.modes))

        return modes_list

    def cleanup(self):
        output_list_of_ops = []
        coeff_counter = 0
        for i in range(len(self.operator_list)):
            if self.operator_list[i] not in output_list_of_ops:
                coeff_counter += self.operator_list[
                    i
                ].coeff  # add coeff of the first operator
                for j in range(i + 1, len(self.operator_list)):
                    if (
                        self.operator_list[i].input_string
                        == self.operator_list[j].input_string
                    ):
                        coeff_counter += self.operator_list[
                            j
                        ].coeff  # add coeffs of other same operators in the list
                output_list_of_ops.append(
                    ParticleOperator(self.operator_list[i].input_string, coeff_counter)
                )
                coeff_counter = 0
        if len(output_list_of_ops) == 1:
            return output_list_of_ops[0]
        else:
            return ParticleOperatorSum(output_list_of_ops)

    def __add__(self, other):
        if isinstance(other, ParticleOperator):
            output = self.operator_list + [other]
            return ParticleOperatorSum(output).cleanup()
        elif isinstance(other, ParticleOperatorSum):
            output = self.operator_list + other.operator_list
            return ParticleOperatorSum(output).cleanup()

    def __mul__(self, other):
        if isinstance(other, Fock):
            out_states = []
            for op in self.operator_list:
                out = op * other
                if isinstance(out, Fock):
                    out_states.append(out)
            if len(out_states) == 1:
                return out_states[0]
            elif len(out_states) == 0:
                return 0
            else:
                return FockSum(out_states)
        elif isinstance(other, FockSum):
            out_states = []
            for op in self.operator_list:
                for state in other.states_list:
                    out = op * state
                    if isinstance(out, Fock):
                        out_states.append(out)
            return FockSum(out_states)

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            ops_w_new_coeffs = []
            for operator in self.operator_list:
                operator.coeff *= other
                ops_w_new_coeffs.append(
                    ParticleOperator(
                        operator.particle_type,
                        operator.modes,
                        operator.ca_string,
                        operator.coeff,
                    )
                )
            return ParticleOperatorSum(ops_w_new_coeffs)


class FermionOperator(ParticleOperator):

    def __init__(self, op, coeff: float = 1.0):
        super().__init__("b" + op)

    def __str__(self):
        return super().__str__()

    def __mul__(self, other):
        return super().__mul__(other)

    def __add__(self, other):
        return super().__add__(other)


class AntifermionOperator(ParticleOperator):
    def __init__(self, op, coeff: float = 1.0):
        super().__init__("d" + op)

    def __str__(self):
        return super().__str__()

    def __mul__(self, other):
        return super().__mul__(other)

    def __add__(self, other):
        return super().__add__(other)


class BosonOperator(ParticleOperator):
    def __init__(self, op, coeff: float = 1.0):
        super().__init__("a" + op)

    def __str__(self):
        return super().__str__()

    def __mul__(self, other):
        return super().__mul__(other)

    def __add__(self, other):
        return super().__add__(other)

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
        self.b_occ = [(n, m) for (n, m) in b_occ if m != 0]
        self.occs = (tuple(f_occ), tuple(af_occ), tuple(b_occ))
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
        # raise NotImplemented
        return None

    def __add__(self, other):
        if isinstance(other, Fock):
            return FockSum([self]).__add__(other)
        elif isinstance(other, FockSum):
            return other.__add__(self)

    def dagger(self):
        return ConjugateFock.from_state(self)


class ConjugateFock:

    def __init__(self, f_occ, af_occ, b_occ, coeff: float = 1.0):
        self.f_occ = f_occ
        self.af_occ = af_occ
        self.b_occ = [(n, m) for (n, m) in b_occ if m != 0]
        self.occs = (tuple(f_occ), tuple(af_occ), tuple(b_occ))
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
            return ConjugateFock(self.f_occ, self.af_occ, self.b_occ, coeff = other*self.coeff)

    def __add__(self, other):
        if isinstance(other, ConjugateFock):
            return ConjugateFockSum([self]).__add__(other)
        elif isinstance(other, ConjugateFockSum):
            return other.__add__(self)

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
            for state in other.FockMap:
                output_value += self * other.FockMap[state][1]
            return output_value
        elif isinstance(other, ParticleOperator):
            # <f|A = (A^dagger |f>)^dagger
            out_state = other.dagger() * self.dagger()
            if isinstance(out_state, (int, float)):
                return out_state
            else:
                new_state = (other.dagger() * self.dagger()).dagger()
                new_state.coeff = other.coeff * self.coeff
                return new_state
                    # other.coeff * self.coeff * (other.dagger() * self.dagger()).dagger()
            
        elif isinstance(other, ParticleOperatorSum):
            # <f|(A + B) = ((A^dagger + B^dagger)|f>)^dagger
            out_state = other.dagger() * self.dagger()
            if isinstance(out_state, (int, float)):
                return out_state
            else:
                out = out_state.dagger()
                return out


class FockSum:
    def __init__(self, states_list):
        if isinstance(states_list, list):
            self.states_list = states_list
            self.FockMap = {}
            self.coeff = states_list[0].coeff
            self.occs = states_list[0].occs
            self.FockMap[self.occs] = (self.coeff, states_list[0])
        elif isinstance(states_list, ConjugateFockSum):
            self.states_list = states_list.states_list
            self.FockMap = states_list.ConjFockMap
            for fock in self.FockMap:
                (coeff, cfock) = self.FockMap[fock]
                self.FockMap[fock] = (coeff, cfock.dagger())
            self.coeff = states_list.coeff
            self.occs = states_list.occs

    def normalize(self):
        normalization = sum([coeff**2 for (coeff, _) in list(self.FockMap.values())])
        return 1.0 / np.sqrt(normalization) * self

    def display(self):
        return display(Latex("$" + self.__str__() + "$"))

    def __str__(self):
        states_str = ""
        for fock in self.FockMap:
            (coeff, state) = self.FockMap[fock]
            f_occ = state.f_occ
            af_occ = state.af_occ
            b_occ = state.b_occ
            states_str += (
                str(coeff)
                + " * |"
                + ",".join([str(i) for i in f_occ][::-1])
                + "; "
                + ",".join([str(i) for i in af_occ][::-1])
                + "; "
                + ",".join([str(i) for i in b_occ][::-1])
                + "⟩"
                + " + "
            )
        return states_str[0:-3]

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            # const. * (|a> + |b>)
            for state in self.FockMap:
                (coeff, fock) = self.FockMap[state]
                self.FockMap[state] = (other * coeff, fock)
            return self

    def __add__(self, other):
        if isinstance(other, Fock):
            if other.occs in self.FockMap:
                # self.FockMap[other.occs] += other.coeff
                other.coeff += self.FockMap[other.occs][0]
                self.FockMap[other.occs] = (other.coeff, other)
            else:
                self.FockMap[other.occs] = (other.coeff, other)
        elif isinstance(other, FockSum):
            for fock in other.FockMap:
                if fock in self.FockMap:
                    (coeff, state) = self.FockMap[fock]
                    self.FockMap[fock] = (coeff + other.FockMap[fock][0], state)
                else:
                    self.FockMap[fock] = other.FockMap[fock]
        return self

    def dagger(self):
        return ConjugateFockSum(self)


class ConjugateFockSum:

    def __init__(self, states_list):
        if isinstance(states_list, list):
            self.states_list = states_list
            self.ConjFockMap = {}
            self.coeff = states_list[0].coeff
            self.occs = states_list[0].occs
            self.ConjFockMap[self.occs] = (self.coeff, states_list[0])
        elif isinstance(states_list, FockSum):
            self.states_list = states_list.states_list
            self.ConjFockMap = states_list.FockMap
            for cfock in self.ConjFockMap:
                (coeff, fock) = self.ConjFockMap[cfock]
                self.ConjFockMap[cfock] = (coeff, fock.dagger())
            self.coeff = states_list.coeff
            self.occs = states_list.occs

    def display(self):
        return display(Latex("$" + self.__str__() + "$"))

    def normalize(self):
        normalization = sum(
            [coeff**2 for (coeff, _) in list(self.ConjFockMap.values())]
        )
        return 1.0 / np.sqrt(normalization) * self

    def __str__(self):
        states_str = ""
        for cfock in self.ConjFockMap:
            (coeff, state) = self.ConjFockMap[cfock]
            f_occ = state.f_occ
            af_occ = state.af_occ
            b_occ = state.b_occ
            states_str += (
                str(coeff)
                + " * ⟨"
                + ",".join([str(i) for i in f_occ][::-1])
                + "; "
                + ",".join([str(i) for i in af_occ][::-1])
                + "; "
                + ",".join([str(i) for i in b_occ][::-1])
                + "|"
                + " + "
            )
        return states_str[0:-3]

    def dagger(self):
        return FockSum(self)

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            # const. * (|a> + |b>)
            for state in self.ConjFockMap:
                (coeff, cfock) = self.ConjFockMap[state]
                self.ConjFockMap[state] = (other * coeff, cfock)
            return self

    def __mul__(self, other):
        if isinstance(other, Fock):
            output_value = 0
            for conj_state in self.ConjFockMap:
                output_value += self.ConjFockMap[conj_state][1] * other
            return output_value
        elif isinstance(other, FockSum):
            output_value = 0
            for conj_state in self.ConjFockMap:
                for state in other.FockMap:
                    output_value += self.ConjFockMap[conj_state][1] * other.FockMap[state][1]
            return output_value
        elif isinstance(other, ParticleOperator):
            list_to_remove = []
            for conj_state in self.ConjFockMap:
                # calculate the new state
                new_state = self.ConjFockMap[conj_state][1] * other
                if isinstance(new_state, (int, float)):
                    list_to_remove.append(conj_state)
                elif isinstance(new_state, ConjugateFock):
                    self.ConjFockMap[conj_state] = (new_state.coeff, new_state)

            # remove the states that have been killed
            for conj_state in list_to_remove:
                del self.ConjFockMap[conj_state]
            return self
        elif isinstance(other, ParticleOperatorSum):
            list_to_remove = []
            for conj_state in self.ConjFockMap:
                for op in other.HashMap:
                    new_state = self.ConjFockMap[conj_state][1] * other.HashMap[op][1]
                    if isinstance(new_state, (int, float)):
                        list_to_remove.append(conj_state)
                    elif isinstance(new_state, ConjugateFock):
                        self.ConjFockMap[conj_state] = (new_state.coeff, new_state)
            for conj_state in list_to_remove:
                if conj_state in self.ConjFockMap:
                    del self.ConjFockMap[conj_state]
            return self

    def __add__(self, other):
        if isinstance(other, ConjugateFock):
            if other.occs in self.ConjFockMap:
                # self.FockMap[other.occs] += other.coeff
                other.coeff += self.ConjFockMap[other.occs][0]
                self.ConjFockMap[other.occs] = (other.coeff, other)
            else:
                self.ConjFockMap[other.occs] = (other.coeff, other)
        elif isinstance(other, ConjugateFockSum):
            for fock in other.ConjFockMap:
                if fock in self.ConjFockMap:
                    (coeff, state) = self.ConjFockMap[fock]
                    self.ConjFockMap[fock] = (coeff + other.ConjFockMap[fock][0], state)
                else:
                    self.ConjFockMap[fock] = other.ConjFockMap[fock]
        return self


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
            orbital = op[1:]
            if orbital[-1] == "^":
                orbital = orbital[:-1]
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
                    op_string += "b^†_{" + str(self.modes[index]) + "}"
                else:
                    op_string += "b_{" + str(self.modes[index]) + "}"

            elif particle == "d":
                antifermion_modes.append(self.modes[index])
                if self.ca_string[index] == "c":
                    op_string += "d^†_{" + str(self.modes[index]) + "}"
                else:
                    op_string += "d_{" + str(self.modes[index]) + "}"

            elif particle == "a":
                boson_modes.append(self.modes[index])
                if self.ca_string[index] == "c":
                    op_string += "a^†_{" + str(self.modes[index]) + "}"
                else:
                    op_string += "a_{" + str(self.modes[index]) + "}"

        self.op_string = str(self.coeff) + " * " + op_string
        self.fermion_modes = fermion_modes
        self.antifermion_modes = antifermion_modes
        self.boson_modes = boson_modes

    def __str__(self):
        return self.op_string

    def __eq__(self, other):
        if isinstance(other, ParticleOperator):
            return self.input_string == other.input_string
        if isinstance(other, ParticleOperatorSum):
            lists_equal = True
            for i in self:
                if i not in other.operator_list:
                    lists_equal = False
            return lists_equal

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
        if isinstance(other, ParticleOperatorSum):
            return other.__add__(self)
        elif isinstance(other, ParticleOperator):
            return ParticleOperatorSum([self]).__add__(other)

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
            for state in other.FockMap:
                state_prime = self.operate_on_state(other.FockMap[state][1])
                if isinstance(state_prime, Fock):
                    other.FockMap[state] = (state_prime.coeff, state_prime)
            return other

    def get_latex(self, op):
        # input_string = op.input_string
        coeff = op.coeff
        len_to_replace = len(str(coeff))
        op.op_string = str(coeff) + op.op_string[len_to_replace:]

        # if 1 <= coeff < 10:
        #     op.op_string = coeff + op_string[3:]
        # elif 10 <= coeff < 100:
        #     op.op_string = coeff + op_string[4:]
        # elif 101 <= coeff < 1000:
        #     op.op_string = coeff + op_string[5:]

        return op.op_string
        

class ParticleOperatorSum:
    # Sum of ParticleOperator instances
    def __init__(self, operator_list: List[ParticleOperator]):
        self.operator_list = operator_list
        self.HashMap = {}
        self.input_string = operator_list[0].input_string
        self.coeff = operator_list[0].coeff
        self.HashMap[self.input_string] = (self.coeff, self.operator_list[0])

                                                            # need to fix __str__!!!!!!the coeff is not right
    def __str__(self):
        op_string = ""
        i = 0
        for op in self.HashMap:
            print("coeff: ", self.HashMap[op][0])
            op_string += self.HashMap[op][1].__str__() + " + "
        return op_string[0:-3]

    def display(self):
        result = ""
        for op in self.HashMap:
            operator = self.HashMap[op][1]
            # need to update the op???????????????????????????????????????????????????
            result += operator.get_latex(operator)
        # display(Latex("$" + self.__str__() + "$"))
        return Latex("$"  + result  + "$")

    def dagger(self):
        out_ops = []
        for op in self.HashMap:
            out_ops.append(self.HashMap[op][1].dagger())

        if len(out_ops) == 1:
            return ParticleOperatorSum(out_ops)
        else:
            result_op_sum = ParticleOperatorSum([out_ops[0]])
            for i, oop in enumerate(out_ops):
                if i == 0:
                    continue
                else:
                    result_op_sum += result_op_sum.__add__(oop)
            return result_op_sum

    def get_modes(self):
        modes_list = []

        for op in self.operator_list:
            modes_list.append(tuple(op.modes))

        return modes_list

    def __add__(self, other):
        if isinstance(other, ParticleOperator):
            if other.input_string in self.HashMap:
                # calc and put in the new coeff
                new_coeff = self.HashMap[other.input_string][0] + other.coeff
                other.coeff = new_coeff
                # insert the new value back
                self.HashMap[other.input_string] = (new_coeff, other)
            else:
                self.HashMap[other.input_string] = (other.coeff, other)
        elif isinstance(other, ParticleOperatorSum):
            for op in other.HashMap:
                if op in self.HashMap:
                    (coeff, operator) = self.HashMap[op]
                    new_coeff = coeff + other.HashMap[op][0]
                    operator.coeff = new_coeff
                    self.HashMap[op] = (new_coeff, operator)
                else:
                    self.HashMap[op] = other.HashMap[op]
        return self

    def __mul__(self, other):
        if isinstance(other, Fock):
            out_states = []
            for op in self.HashMap:
                out = self.HashMap[op][1] * other
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
            # for op in self.operator_list:
            for op in self.HashMap:
                for state in other.states_list:
                    out = self.HashMap[op][1] * state
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

    def __eq__(self, other):
        if isinstance(other, ParticleOperatorSum):
            lists_equal = True
            for i in self.operator_list:
                if i not in other.operator_list:
                    lists_equal = False
            return lists_equal


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

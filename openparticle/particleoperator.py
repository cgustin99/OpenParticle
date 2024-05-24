import numpy as np
from sympy import *
from typing import List
from openparticle.fock import *
from IPython.display import display, Latex

class ParticleOperator():

    def __init__(self, input_string, coeff = 1.0):
        #Initializes particle operator in the form of b_n d_n a_n
        #Parameters:
        #input_string: e.g. 'b2^ a0'
        
        self.input_string = input_string
        self.coeff = coeff

        particle_type = ''
        modes = []
        ca_string = ''

        for op in self.input_string.split(" ")[::-1]:
            type = op[0]
            orbital = op[1]
            particle_type += type
            modes.append(int(orbital))
            if op[-1] != '^':
                ca_string += "a"
            else: ca_string += 'c'

        self.particle_type = particle_type
        self.modes = modes
        self.ca_string = ca_string

        fermion_modes = []
        antifermion_modes = []
        boson_modes = []

        op_string = ''
        for index, particle in enumerate(self.particle_type):

            if particle == 'b':
                fermion_modes.append(self.modes[index])
                if self.ca_string[index] == 'c':
                    op_string += 'b^†_' + str(self.modes[index])
                else:
                    op_string += 'b_' + str(self.modes[index])

            elif particle == 'd': 
                antifermion_modes.append(self.modes[index])
                if self.ca_string[index] == 'c':
                    op_string += 'd^†_' + str(self.modes[index])
                else: 
                    op_string += 'd_' + str(self.modes[index])

            elif particle == 'a': 
                boson_modes.append(self.modes[index])
                if self.ca_string[index] == 'c':
                    op_string += 'a^†_' + str(self.modes[index])
                else: 
                    op_string += 'a_' + str(self.modes[index]) 

        self.op_string = str(self.coeff) + '*' + op_string
        self.fermion_modes = fermion_modes
        self.antifermion_modes = antifermion_modes
        self.boson_modes = boson_modes

    def __str__(self):
        return self.op_string

    
    def display(self):
        display(Latex('$' + self.op_string + '$'))
        
    def __add__(self, other):
        if self.input_string == other.input_string:
            return ParticleOperator(self.input_string, self.coeff + other.coeff)
        else:
            return ParticleOperatorSum([self, other])

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            self.coeff *= other
            return ParticleOperator(self.input_string, self.coeff)

    def impose_parity_jw(self, state, mode):
            index = state.index(mode)
            parity_str = state[:index]
            coeff = (-1)**len(parity_str)
            return coeff

    def operate_on_state(self, other):
        coeff = self.coeff * other.coeff
        updated_ferm_state = other.f_occ[:]
        updated_antiferm_state = other.af_occ[:]
        updated_bos_state = other.b_occ[:]
        
        for op in self.input_string.split(" "):
            if op[-1] == '^':
                if op[0] == 'b':
                    if int(op[1]) not in other.f_occ:
                        updated_ferm_state.append(int(op[1]))
                        coeff *= self.impose_parity_jw(updated_ferm_state, int(op[1]))
                    else: coeff = 0
                elif op[0] == 'd':
                    if int(op[1]) not in other.af_occ:
                        updated_antiferm_state.append(int(op[1]))
                        coeff *= self.impose_parity_jw(updated_antiferm_state, int(op[1]))
                    else: coeff = 0
                elif op[0] == 'a':
                    state_modes, state_occupancies = [i[0] for i in other.b_occ], [i[1] for i in other.b_occ]
                    
                    if int(op[1]) in state_modes:
                        index = state_modes.index(int(op[1]))
                        if state_occupancies[index] >= 1:
                            state_occupancies[index] += 1
                            coeff *= np.sqrt(state_occupancies[index])
                        
                    else:
                        state_modes.append(int(op[1]))
                        state_occupancies.append(1)
                    #zip up modes and occupancies into an updated list
                    updated_bos_state = list(zip(state_modes, state_occupancies))
                    sorted_updated_bos_state = sorted(updated_bos_state, key=lambda x: x[0])
            else:
                if op[0] == 'b':
                    if int(op[1]) in other.f_occ:
                        coeff *= self.impose_parity_jw(updated_ferm_state, int(op[1]))
                        updated_ferm_state.remove(int(op[1]))
                        
                    else: coeff = 0
                elif op[0] == 'd':
                    if int(op[1]) in other.af_occ:
                        coeff *= self.impose_parity_jw(updated_antiferm_state, int(op[1]))
                        updated_antiferm_state.remove(int(op[1]))
                    else: coeff = 0
                elif op[0] == 'a':
                    state_modes, state_occupancies = [i[0] for i in other.b_occ], [i[1] for i in other.b_occ]
                    
                    if int(op[1]) in state_modes:
                        index = state_modes.index(int(op[1]))
                        if state_occupancies[index] > 1:
                            state_occupancies[index] -= 1
                            coeff *= np.sqrt(state_occupancies[index] + 1)
                        else:
                            #state_modes.remove(int(op[1]))
                            #state_occupancies.remove(int(op[1]))
                            coeff = 0
                    else: coeff = 0
                    #zip up modes and occupancies into an updated list
                    updated_bos_state = list(zip(state_modes, state_occupancies))
                    sorted_updated_bos_state = sorted(updated_bos_state, key=lambda x: x[0])
        
        if 'a' in self.particle_type:
            return coeff * Fock(sorted(updated_ferm_state),
                                 sorted(updated_antiferm_state), sorted_updated_bos_state)
        else: return coeff * Fock(sorted(updated_ferm_state),
                                 sorted(updated_antiferm_state), updated_bos_state)

        
    def __mul__(self, other):
        if isinstance(other, ParticleOperator):
            updated_coeff = self.coeff * other.coeff
            return ParticleOperator(self.input_string + " " + other.input_string, updated_coeff)

        elif isinstance(other, FockSum):
            updated_states = []
            for state in other.states_list:
                state_prime = self.operate_on_state(state)
                if isinstance(state_prime, Fock):
                    updated_states.append(state_prime)
            return FockSum(updated_states)
        
        elif isinstance(other, Fock):
            return self.operate_on_state(other)
            

class ParticleOperatorSum():
    #Sum of ParticleOperator instances
    def __init__(self, operator_list: List[ParticleOperator]):
        self.operator_list = operator_list    

    def __str__(self):
        op_string = ''
        for index, op in enumerate(self.operator_list):
            if index != len(self.operator_list) - 1:
                op_string += op.__str__() + " + "
            else: op_string += op.__str__()
        return op_string
    
    def display(self):
        return display(Latex('$' + self.__str__() + '$'))

    def __add__(self, other):
        if isinstance(other, ParticleOperator):
            for op in range(len(self.operator_list)):
                if self.operator_list[op].input_string == other.input_string:
                    self.operator_list[op] = ParticleOperator(self.operator_list[op].input_string, other.coeff)
                    break
                else:
                    self.operator_list.append(other)
                    break

        return ParticleOperatorSum(self.operator_list)
    
    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            ops_w_new_coeffs = []
            for operator in self.operator_list:
                operator.coeff *= other
                ops_w_new_coeffs.append(ParticleOperator(operator.particle_type, 
                            operator.modes, operator.ca_string, operator.coeff))
            return ParticleOperatorSum(ops_w_new_coeffs)

class FermionOperator(ParticleOperator):

    def __init__(self, op, coeff: float = 1.0):
        super().__init__('b' + op)
        
    def __str__(self):
        return super().__str__()
    
    def __mul__(self, other):
        return super().__mul__(other)
    
    def __add__(self, other):
        return super().__add__(other)
  
class AntifermionOperator(ParticleOperator):
    def __init__(self, op, coeff: float = 1.0):
        super().__init__('d' + op)
        
    def __str__(self):
        return super().__str__()
    
    def __mul__(self, other):
        return super().__mul__(other)
    
    def __add__(self, other):
        return super().__add__(other)

class BosonOperator(ParticleOperator):
    def __init__(self, op, coeff: float = 1.0):
        super().__init__('a' + op)
        
    def __str__(self):
        return super().__str__()
    
    def __mul__(self, other):
        return super().__mul__(other)
    
    def __add__(self, other):
        return super().__add__(other)



import numpy as np
from sympy import *
from typing import List
from IPython.display import display, Latex

class ParticleOperator():

    def __init__(self, op_str, coeff = 1.0):
        self.op_str = op_str
        self.coeff = coeff

        particle_type = ''
        modes = []
        ca_string = ''

        for op in self.op_str.split(" "):
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

        op_string = ''
        for index, particle in enumerate(self.particle_type):

            if particle == 'b':
                if self.ca_string[index] == 'c':
                    op_string += 'b^†_' + str(self.modes[index])
                else:
                    op_string += 'b_' + str(self.modes[index])

            elif particle == 'd': 
                if self.ca_string[index] == 'c':
                    op_string += 'd^†_' + str(self.modes[index])
                else: 
                    op_string += 'd_' + str(self.modes[index])

            elif particle == 'a': 
                if self.ca_string[index] == 'c':
                    op_string += 'a^†_' + str(self.modes[index])
                else: 
                    op_string += 'a_' + str(self.modes[index]) 

        self.op_string = str(self.coeff) + '*' + op_string

    def __str__(self):
        return self.op_string

    
    def display(self):
        display(Latex('$' + self.op_string + '$'))
        
    def __add__(self, other):
        return ParticleOperatorSum([self, other])

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            self.coeff *= other
            return ParticleOperator(self.particle_type, self.modes, self.ca_string, self.coeff)

    def impose_parity_jw(self, state, mode):
            index = state.index(mode)
            parity_str = state[:index]
            coeff = (-1)**len(parity_str)
            return coeff
        
    def __mul__(self, other):
        if isinstance(other, ParticleOperator):
            updated_coeff = self.coeff * other.coeff
            return ParticleOperator(self.op_str + " " + other.op_str, updated_coeff)
        
        elif isinstance(other, FockState):
            coeff = self.coeff * other.coeff
            updated_ferm_state = other.f_occ[:]
            updated_antiferm_state = other.af_occ[:]
            updated_bos_state = other.b_occ[:]
            
            for op in self.op_str.split(" "):
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
                return coeff * FockState(sorted(updated_ferm_state),
                                     sorted(updated_antiferm_state), sorted_updated_bos_state)
            else: return coeff * FockState(sorted(updated_ferm_state),
                                     sorted(updated_antiferm_state), updated_bos_state)
        
            

class ParticleOperatorSum(ParticleOperator):

    def __init__(self, operator_list: List[ParticleOperator]):
        self.operator_list = operator_list    

    def __str__(self):
        op_str = ''
        for index, op in enumerate(self.operator_list):
            if index != len(self.operator_list) - 1:
                op_str += op.__str__() + " + "
            else: op_str += op.__str__()
        return op_str
    
    def display(self):
        return display(Latex('$' + self.__str__() + '$'))
    
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

class FockState():

    '''
    Defines a Fock state of the form |f, \bar{f}, b⟩

    Parameters:
    ferm_occupancy, antiferm_occupancy, bos_occupancy: List 
    e.g. a hadron with a fermion in mode 2, an antifermion in mode 1 and 2 bosons
    in mode 1 is 
    ferm_occupancy = [2]
    antiferm_occupancy = [1]
    bos_occupancy = [(1, 2)]
    
    '''

    def __init__(self, f_occ, af_occ, b_occ, coeff: float = 1.0):
        self.f_occ = f_occ
        self.af_occ = af_occ
        self.b_occ = b_occ
        self.coeff = coeff

    def __str__(self):
        return str(self.coeff) + " * |" + ",".join([str(i) for i in self.f_occ]) + "; " +\
            ",".join([str(i) for i in self.af_occ]) + "; " +\
            ",".join([str(i) for i in self.b_occ]) + "⟩"
    
    def display(self):
        display(Latex('$' + self.__str__() + '$'))


    def __rmul__(self, other):
        if isinstance(other,(float, int)):
            if other == 0:
                return 0
            else:
                coeff = self.coeff * other
                return FockState(self.f_occ, self.af_occ, self.b_occ, coeff)
        elif isinstance(other, ConjugateFockState):
            raise NotImplemented
        elif isinstance(other, FockState):
            raise Exception("Cannot multiply a ket to the left of a ket")
        
    def __mul__(self, other):
        raise NotImplemented

    
    def __add__(self, other):
        return FockStateSum([self, other])
    
    def dagger(self):
        return ConjugateFockState.from_state(self)


class ConjugateFockState(FockState): 

    def __init__(self, f_occ, af_occ, b_occ, coeff: float = 1.0): 
        self.f_occ = f_occ
        self.af_occ = af_occ
        self.b_occ = b_occ
        self.coeff = coeff

    def __str__(self):
        return str(self.coeff) + " * " + "⟨" + ",".join([str(i) for i in self.f_occ]) + "; " +\
            ",".join([str(i) for i in self.af_occ]) + "; " +\
            ",".join([str(i) for i in self.b_occ]) + "|"
    
    @classmethod
    def from_state(cls, state: FockState):
        f_occ = state.f_occ
        af_occ = state.af_occ
        b_occ = state.b_occ
        coeff = state.coeff
        return cls(f_occ, af_occ, b_occ, coeff)
    
    def display(self):
        display(Latex('$' + self.__str__() + '$'))
    
    def inner_product(self, other):
        if self.f_occ == other.f_occ and \
            self.af_occ == other.af_occ and \
            self.b_occ == other.b_occ: 
            return 1.0
        else: return 0

    def __mul__(self, other):
        if isinstance(other, FockState):
            return self.inner_product(other)

    
class FockStateSum(FockState):
    def __init__(self, states_list: List[FockState]):
        self.states_list = states_list

    def normalize(self):
        #TODO
        return

    def __str__(self):
        states_str = ''
        for index, state in enumerate(self.states_list):
            if index != len(self.states_list) - 1:
                states_str += state.__str__() + " + "
            else: states_str += state.__str__()

        return states_str
    
    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            for state in self.states_list:
                state.coeff *= other
            return self
    

import numpy as np
from sympy import *
from typing import List
from FockState import *
from IPython.display import display, Latex

class ParticleOperator():

    def __init__(self, particle_str, modes, ca_string, coeff: float = 1.0):
        self.particle_str = particle_str
        self.modes = modes
        self.ca_string = ca_string
        self.coeff = coeff
        self.info = np.array([particle_str, modes, ca_string], dtype = object)

        op_string = ''
        for index, particle in enumerate(self.particle_str):

            if particle == 'f':
                if self.ca_string[index] == 'c':
                    op_string += 'b^†_' + str(self.modes[index])
                else:
                    op_string += 'b_' + str(self.modes[index])

            elif particle == 'a': 
                if self.ca_string[index] == 'c':
                    op_string += 'd^†_' + str(self.modes[index])
                else: 
                    op_string += 'd_' + str(self.modes[index])

            elif particle == 'b': 
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
            return ParticleOperator(self.particle_str, self.modes, self.ca_string, self.coeff)

    def __mul__(self, other):
        if isinstance(other, ParticleOperator):
            updated_particle_str = self.particle_str + other.particle_str
            updated_modes = self.modes + other.modes
            updated_ca_string = self.ca_string + other.ca_string
            updated_coeff = self.coeff * other.coeff
            return ParticleOperator(updated_particle_str, updated_modes, updated_ca_string, updated_coeff)
        
        elif isinstance(other, FockState):
            
            updated_ferm_occupancy = other.ferm_occupancy[:]
            updated_antiferm_occupancy = other.antiferm_occupancy[:]
            updated_bos_occupancy = other.bos_occupancy[:]

            coeff = other.coeff
            
            for mode in range(len(self.modes)):
                if self.particle_str[mode] == 'f':
                    if self.ca_string[mode] == 'a':
                        if other.ferm_occupancy[self.modes[mode]] == 0:
                            return 0
                        else: 
                            updated_ferm_occupancy[self.modes[mode]] = 0
                            coeff *= (-1)**np.sum(other.ferm_occupancy[0:self.modes[mode]])
                    elif self.ca_string[mode] == 'c':
                        if other.ferm_occupancy[self.modes[mode]] == 0:
                            updated_ferm_occupancy[self.modes[mode]] = 1
                            coeff *= (-1)**np.sum(other.ferm_occupancy[0:self.modes[mode]])
                        else: return 0
                elif self.particle_str[mode] == 'a':
                    if self.ca_string[mode] == 'a':
                        if other.antiferm_occupancy[self.modes[mode]] == 0:
                            return 0
                        else: 
                            updated_antiferm_occupancy[self.modes[mode]] = 0
                            coeff *= (-1)**np.sum(other.antiferm_occupancy[0:self.modes[mode]])
                    elif self.ca_string[mode] == 'c':
                        if other.antiferm_occupancy[self.modes[mode]] == 0:
                            updated_antiferm_occupancy[self.modes[mode]] = 1
                            coeff *= (-1)**np.sum(other.antiferm_occupancy[0:self.modes[mode]])
                        else: return 0
                elif self.particle_str[mode] == 'b':
                    if self.ca_string[mode] == 'a':
                        if other.bos_occupancy[self.modes[mode]] == 0:
                            return 0
                        else: 
                            updated_bos_occupancy[self.modes[mode]] = other.bos_occupancy[self.modes[mode]] - 1
                            coeff *= np.sqrt(other.bos_occupancy[mode])
                    elif self.ca_string[mode] == 'c':
                        updated_bos_occupancy[self.modes[mode]] = other.bos_occupancy[self.modes[mode]] + 1
                        coeff *= np.sqrt(other.bos_occupancy[mode] + 1)
        

        return FockState(updated_ferm_occupancy, updated_antiferm_occupancy, updated_bos_occupancy, coeff)

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
                ops_w_new_coeffs.append(ParticleOperator(operator.particle_str, 
                            operator.modes, operator.ca_string, operator.coeff))
            return ParticleOperatorSum(ops_w_new_coeffs)

class FermionOperator(ParticleOperator):

    def __init__(self, mode, ca, coeff: float = 1.0):
        super().__init__('f', [mode], ca, coeff)
        
    def __str__(self):
        return super().__str__()
    
    def __mul__(self, other):
        return super().__mul__(other)
    
    def __add__(self, other):
        return super().__add__(other)
  
class AntifermionOperator(ParticleOperator):
    def __init__(self, mode, ca, coeff: float = 1.0):
        super().__init__('a', [mode], ca, coeff)
        
    def __str__(self):
        return super().__str__()
    
    def __mul__(self, other):
        return super().__mul__(other)
    
    def __add__(self, other):
        return super().__add__(other)

class BosonOperator(ParticleOperator):
    def __init__(self, mode, ca, coeff: float = 1.0):
        super().__init__('b', [mode], ca, coeff)
        
    def __str__(self):
        return super().__str__()
    
    def __mul__(self, other):
        return super().__mul__(other)
    
    def __add__(self, other):
        return super().__add__(other)



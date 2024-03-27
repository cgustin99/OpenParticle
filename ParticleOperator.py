import numpy as np
from sympy import *
from typing import List
from FockState import *

class ParticleOperator():

    def __init__(self, particle_list, modes, ca_string, coeff = 1.0):
        self.particle_list = particle_list
        self.modes = modes
        self.ca_string = ca_string
        self.coeff = coeff
        self.info = np.array([particle_list, modes, ca_string], dtype = object)

    def __str__(self):
        op_string = ""

        for index, particle in enumerate(self.particle_list):
            
            if particle == 'f':
                if self.ca_string[index] == 'c':
                    op_string += "b†(" + str(self.modes[index]) + ")"
                else: 
                    op_string += "b(" + str(self.modes[index]) + ")"
            elif particle == 'a': 
                if self.ca_string[index] == 'c':
                    op_string += "d†(" + str(self.modes[index]) + ")"
                else: 
                    op_string += "d(" + str(self.modes[index]) + ")"
            elif particle == 'b':
                if self.ca_string[index] == 'c':
                    op_string += "a†(" + str(self.modes[index]) + ")"
                else: op_string += "a(" + str(self.modes[index]) + ")"

        return str(self.coeff) + " * " + op_string
    
    def __add__(self, other):
        return ParticleOperatorSum([self, other])

    def __rmul__(self, other: (float, int)):
        self.coeff = other
        return self

    def __mul__(self, other):
        if isinstance(other, ParticleOperator):
            updated_particle_list = self.particle_list + other.particle_list
            updated_modes = self.modes + other.modes
            updated_ca_string = self.ca_string + other.ca_string
            return ParticleOperator(updated_particle_list, updated_modes, updated_ca_string)
        
        elif isinstance(other, FockState):
            
            updated_ferm_occupancy = other.ferm_occupancy[:]
            updated_antiferm_occupancy = other.antiferm_occupancy[:]
            updated_bos_occupancy = other.bos_occupancy[:]

            coeff = other.coeff
            
            for mode in range(len(self.modes)):
                if self.particle_list[mode] == 'f':
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
                elif self.particle_list[mode] == 'a':
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
                elif self.particle_list[mode] == 'b':
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

class FermionOperator(ParticleOperator):

    def __init__(self, mode, ca):
        super().__init__('f', [mode], ca)

    def __str__(self):
        if self.ca_string == 'a':
            return "b(" + str(self.modes[0]) + ")"
        elif self.ca_string == "c":
            return "b†(" + str(self.modes[0]) + ")"
  

class AntifermionOperator(ParticleOperator):
    def __init__(self, mode, ca):
        super().__init__('a', [mode], ca)

    def __str__(self):
        if self.ca_string == 'a':
            return "d(" + str(self.modes[0]) + ")"
        elif self.ca_string == "c":
            return "d†(" + str(self.modes[0]) + ")"
    
class BosonOperator(ParticleOperator):
    def __init__(self, mode, ca):
        super().__init__('b', [mode], ca)

    def __str__(self):
        if self.ca_string == 'a':
            return "a(" + str(self.modes[0]) + ")"
        elif self.ca_string == "c":
            return "a†(" + str(self.modes[0]) + ")"



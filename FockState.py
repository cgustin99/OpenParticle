import numpy as np
from ParticleOperator import *
from typing import List


class FockState():

    def __init__(self, ferm_occupancy, antiferm_occupancy, bos_occupancy, coeff):
        self.ferm_occupancy = ferm_occupancy
        self.antiferm_occupancy = antiferm_occupancy
        self.bos_occupancy = bos_occupancy
        self.coeff = coeff
    

    def __str__(self):
        fock_str = str(self.coeff) + " * " + "|" + ",".join([str(i) for i in self.ferm_occupancy]) + "; " +\
            ",".join([str(i) for i in self.antiferm_occupancy]) + "; " +\
            ",".join([str(i) for i in self.bos_occupancy]) + "⟩"
       

        return fock_str


    def __rmul__(self, other):
        if isinstance(other,(float, int)):
            self.coeff = other
            return self
        elif isinstance(other, ConjugateFockState):
            raise NotImplemented
        elif isinstance(other, FockState):
            raise Exception("Cannot multiply a ket to the left of a ket")
        
    def __mul__(self, other):
        raise NotImplemented

    def compact_state(self):

        num_nonzero = 0

        ferm_compact = np.nonzero(self.ferm_occupancy)[0]
        antiferm_compact = np.nonzero(self.antiferm_occupancy)[0]
        bos_compact = list(zip(np.nonzero(self.bos_occupancy)[0],
                       np.array(self.bos_occupancy)[np.nonzero(self.bos_occupancy)[0]]))
        
        fock_str = "|" + ",".join([str(i) for i in ferm_compact]) + "; " +\
            ",".join([str(i) for i in antiferm_compact]) + "; " +\
            ",".join([str(i) for i in bos_compact]) + "⟩"
        
        return fock_str
    
    def __add__(self, other):
        return FockStateSum([self, other])
    
    def dagger(self):
        return ConjugateFockState(self)

class ConjugateFockState(FockState):
    def __init__(self, state: FockState):
        self.state = state

    def __str__(self):
        fock_str = str(self.state.coeff) + " * " + "⟨" + ",".join([str(i) for i in self.state.ferm_occupancy]) + "; " +\
            ",".join([str(i) for i in self.state.antiferm_occupancy]) + "; " +\
            ",".join([str(i) for i in self.state.bos_occupancy]) + "|"
        return fock_str
    
    def inner_product(self, other):
        # print(self.state.ferm_occupancy, other.ferm_occupancy)
        if self.state.ferm_occupancy == other.ferm_occupancy and \
            self.state.antiferm_occupancy == other.antiferm_occupancy and \
            self.state.bos_occupancy == other.bos_occupancy: 
            return 1.0
        else: return 0

    def __mul__(self, other):
        if isinstance(other, FockState):
            return self.inner_product(other)
    
class FockStateSum(FockState):
    def __init__(self, states_list: List[FockState]):
        self.states_list = states_list

    def __str__(self):
        states_str = ''
        for index, state in enumerate(self.states_list):
            if index != len(self.states_list) - 1:
                states_str += state.__str__() + " + "
            else: states_str += state.__str__()

        return states_str
    
    
   

    

    

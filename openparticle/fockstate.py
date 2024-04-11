import numpy as np
from typing import List
from IPython.display import display, Latex


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
        return str(self.coeff) + " * |" + ",".join([str(i) for i in self.f_occ][::-1]) + "; " +\
            ",".join([str(i) for i in self.af_occ][::-1]) + "; " +\
            ",".join([str(i) for i in self.b_occ][::-1]) + "⟩"
    
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
        return str(self.coeff) + " * " + "⟨" + ",".join([str(i) for i in self.f_occ][::-1]) + "; " +\
            ",".join([str(i) for i in self.af_occ][::-1]) + "; " +\
            ",".join([str(i) for i in self.b_occ][::-1]) + "|"
    
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
    
    
   

    

    

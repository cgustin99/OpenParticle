from QCD.get_quantum_nums import *
from openparticle import FockState, FockStateSum, ParticleOperator
from itertools import product
from IPython.display import display, Latex
from collections import Counter


class Hadron():

    def __init__(self, sectors):
        self.sectors = sectors
        self.particle_type = self.get_particle_type()


    def get_particle_type(self):
        if self.sectors[0].count('q') == self.sectors[0].count('qbar'):
            return "meson"
        elif self.sectors[0].count('q') != self.sectors[0].count('qbar'):
            return "baryon"
        
    def __str__(self):
        state_str = "|"

        for index, state in enumerate(self.sectors):
            for particle in state:
                if particle == 'q':
                    state_str += 'q'
                elif particle == 'qbar':
                    state_str += '\\bar{q}'
                elif particle == 'g':
                    state_str += 'g'
            if index == len(self.sectors) - 1:
                state_str += ">"
            else: state_str += "> + |"
                
        return state_str
    
    def display(self):
        return display(Latex('$' + self.__str__() + '$'))
    
    def get_states(self, K):
        qnums = get_quantum_numbers(K)
        
        quarks = qnums[qnums['n'] % 2 == 1]
        antiquarks = qnums[qnums['n'] % 2 == 1]
        gluons = qnums[qnums['n'] % 2 == 0]

        valid_states = []

        for sector in self.sectors:
            fermi_indices = []
            antifermi_indices = []
            bose_indices = []
            
            #quarks
            for pair in itertools.combinations(quarks.index.to_numpy(), r = sector.count('q')):
                fermi_indices.append(pair)
            
            #antiquarks
            for pair in itertools.combinations(quarks.index.to_numpy(), r = sector.count('qbar')):
                antifermi_indices.append(pair)

            #gluons
            for pair in itertools.combinations_with_replacement(gluons.index.to_numpy(), r = sector.count('g')):
                bose_indices.append(pair)
            
            #Check momentum sums 
            for pair in itertools.product(fermi_indices, antifermi_indices, bose_indices):
                nested_lists = [list(inner_tuple) for inner_tuple in pair]
                indices = [item for sublist in nested_lists for item in sublist]
                if qnums.loc[indices, 'n'].sum() == K:
                    nested_lists_full = self.pad_states(nested_lists, len(qnums))
                    valid_states.append(FockState(nested_lists_full[0], 
                                                  nested_lists_full[1], 
                                                  nested_lists_full[2]))
                    
        return FockStateSum(valid_states)
    

    def pad_states(self, state, num_modes):
        #pad quarks
        padded_quarks = [1 if i in state[0] else 0 for i in range(num_modes)]
        if padded_quarks == [0] * num_modes:
            padded_quarks = []

        #pad_antiquarks
        padded_antiquarks = [1 if i in state[1] else 0 for i in range(num_modes)]
        if padded_antiquarks == [0] * num_modes:
            padded_antiquarks = []

        #pad gluons
        index_counts = Counter(state[2])
        padded_gluons = [index_counts[i] if i in index_counts else 0 for i in range(num_modes)]
        if padded_gluons == [0] * num_modes:
            padded_gluons = []

        

        return [padded_quarks, padded_antiquarks, padded_gluons]

            

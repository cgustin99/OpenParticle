from openparticle.QCD import get_quantum_numbers
import numpy as np
from itertools import product
from IPython.display import Latex, display
import itertools
from openparticle.fockstate import FockState, FockStateSum
from collections import Counter


class Hadron():

    def __init__(self, particle_type, K):
        self.particle_type = particle_type
        self.K = K
        self.sectors = self.get_sectors(K)

    def get_sectors(self, K):
        if self.particle_type == 'meson':
            sectors = []
            for k in range(1, K + 1):
                sectors_set = set(product('Qg', repeat = k))
                sectors_unique = set()
                for comb in sectors_set:
                    sectors_unique.add( ''.join(sorted(comb)) )
                sectors += [[item] for item in sectors_unique]

            processed_sectors = []
            for sec in sectors:
                string_list = [* sec[0]]
                new_list = []
                for item in string_list:
                    if item == 'Q':
                        new_list.extend(['q', 'qbar'])
                    else:
                        new_list.append(item)
                processed_sectors.append(new_list)
            return processed_sectors
                
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
                state_str += "⟩"
            else: state_str += "⟩ + |"
                
        return state_str
    

    def display(self):
        return display(Latex('$' + self.__str__() + '$'))
    
    def get_states(self, K, P_perp, Lambda_perp):
        assert K == self.K
        
        qnums = get_quantum_numbers(K, Lambda_perp)
        
        quarks = qnums[qnums['n'] % 1 == 0.5]#can reindex here with .reset_index()
        antiquarks = qnums[qnums['n'] % 1 == 0.5]
        gluons = qnums[qnums['n'] % 1 == 0]


        valid_states = []

        for sector in self.sectors:
            fermi_indices = []
            antifermi_indices = []
            bose_indices = []
            
            #quarks
            for pair in itertools.combinations(quarks.index.to_numpy(), r = sector.count('q')):
                fermi_indices.append(pair)
            
            #antiquarks
            for pair in itertools.combinations(antiquarks.index.to_numpy(), r = sector.count('qbar')):
                antifermi_indices.append(pair)

            #gluons
            for pair in itertools.combinations_with_replacement(gluons.index.to_numpy(), r = sector.count('g')):
                bose_indices.append(pair)
            
            #Check momentum sums 
            for pair in itertools.product(fermi_indices, antifermi_indices, bose_indices):
                nested_lists = [list(inner_tuple) for inner_tuple in pair]
                indices = [item for sublist in nested_lists for item in sublist]
                if qnums.loc[indices, 'n'].sum() == K and qnums.loc[indices, 'n⟂'].sum() == P_perp:
                    valid_states.append(FockState(nested_lists[0], nested_lists[1], self.postprocess_bosons(nested_lists[2])))
                  
                    
        return FockStateSum(valid_states)
    
    def postprocess_bosons(self, boson_state):
        counter = Counter(sorted(boson_state))
        result = list(counter.items())
        return result
    

    


            

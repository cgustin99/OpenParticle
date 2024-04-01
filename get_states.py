import numpy as np
import pandas as pd
from typing import Union
import itertools

colors = np.array(['r', 'g', 'b'])
helicities = np.array(['+1', '-1'])
flavors = np.array(['u', 'd'])#SU(2) flavor, 't', 'b', 's', 'c'])


def get_quantum_numbers(K: Union[float, int], Lambda_perp: Union[float, int],
               data_frame: bool = False, impose_singlets: bool = True ):
    perp_modes = np.arange(-Lambda_perp, Lambda_perp + 1)
    longitudinal_modes = np.arange(1, K)

    quantum_numbers = [helicities, longitudinal_modes, perp_modes]
    q_states = list(itertools.product(*quantum_numbers))

    if data_frame: 
        if impose_singlets:
            return pd.DataFrame(q_states, columns = ['Helicity', 'n', 'n⟂'])
    
    else: return q_states

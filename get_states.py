import numpy as np
import pandas as pd
from typing import Union
import itertools

colors = np.array(['r', 'g', 'b'])
helicities = np.array(['+1', '-1'])
flavors = np.array(['u', 'd', 't', 'b', 's', 'c'])


def get_states(K: Union[float, int], Lambda_perp: Union[float, int],
               data_frame: bool = False ):
    perp_modes = np.arange(-Lambda_perp, Lambda_perp + 1)
    longitudinal_modes = np.arange(1, K)

    quantum_numbers = [colors, helicities, flavors, longitudinal_modes, perp_modes]
    q_states = list(itertools.product(*quantum_numbers))

    if data_frame: 
        return pd.DataFrame(q_states, columns = ['Color', 'Helicity', 'Flavor', 'n', 'n⟂'])
    
    else: return q_states

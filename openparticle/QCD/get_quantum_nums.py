import numpy as np
import pandas as pd
from typing import Union
import itertools

colors = np.array(['r', 'g', 'b'])
helicities = np.array(['+1', '-1'])
flavors = np.array(['u', 'd'])#SU(2) flavor, 't', 'b', 's', 'c'])

def get_quantum_numbers(K: Union[float, int], P_perp: Union[float, int]):

    longitudinal_modes = np.arange(1/2, K + 1/2, 1/2)
    perp_modes = np.arange(-P_perp, P_perp + 1)

    quantum_numbers = [helicities, longitudinal_modes, perp_modes]
    q_states = list(itertools.product(*quantum_numbers))

    qnums = pd.DataFrame(q_states, columns = ['Helicity', 'n', 'n⟂'])
    
    return qnums


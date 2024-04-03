import numpy as np

#SU(3) generators T^a = \lambda^a/2 satisfy Lie Algebra
#[Ta, Tb] = if^abc Tc

def get_color_factor(c, cprime, a, conjugate = False):

    lam = np.array([np.array([[0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 0]]),
            np.array([[0, -1j, 0],
                    [1j, 0, 0],
                    [0, 0, 0]]),
            np.array([[1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 0]]),
            np.array([[0, 0, 1],
                    [0, 0, 0],
                    [1, 0, 0]]),
            np.array([[0, 0, -1j],
                    [0, 0, 0],
                    [1j, 0, 0]]),
            np.array([[0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0]]),
            np.array([[0, 0, 0],
                    [0, 0, -1j],
                    [0, 1j, 0]]),  
            np.array([[1/np.sqrt(3), 0, 0],
                    [0, 1/np.sqrt(3), 0],
                    [0, 0, -2/np.sqrt(3)]])

])
    T = lam/2
    
    chis = np.array([np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])])
    chis_columns = chis.reshape(3,3)

    if conjugate == True: return chis_columns[:,cprime].conj().T @ T[a].conj().T @ chis_columns[:,c]
    else: return chis_columns[:,cprime].conj().T @ T[a] @ chis_columns[:,c]

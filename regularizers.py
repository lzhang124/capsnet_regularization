import numpy as np
from numpy import linalg as LA

def combined_regularizer(regularizers, weights):
    def reg_fn(M):
        loss= 0
        for i in range(len(regularizers)):
            loss += weights[i] * regularizers[i](M)          
        return loss

    return reg_fn


def l2PerRow(M):
    d1, d2, d3, d4 = M.shape
    flatM = np.reshape(d1*d2, d3*d4)
    return np.sum(np.square(LA.norm(flatM, axis=1)))


def frobenius(M):
    d1, d2, d3, d4 = M.shape
    entryPerW = np.reshape(d1*d2, d3, d4)
    return np.sum(LA.norm(entryPerW, 'fro'))


def operatorNorm(M):
    d1, d2, d3, d4 = M.shape
    entryPerW = np.reshape(d1*d2, d3, d4)
    # ord 2 is operator norm: https://stackoverflow.com/questions/48275198/how-does-numpy-linalg-norm-ord-2-work
    return np.sum(LA.norm(entryPerW, 2)) 


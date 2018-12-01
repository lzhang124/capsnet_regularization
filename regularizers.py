from keras import backend as K
import tensorflow as tf

def combined_regularizer(regularizers, weights):
    if len(regularizers) == 0:
        return None

    assert len(regularizers) == len(weights)
    
    def reg_fn(M):
        loss = 0
        for i in range(len(regularizers)):
            loss += weights[i] * regularizers[i](M)          
        return loss
    return reg_fn


def l2_row(M):
    d1, d2, d3, d4 = map(int, M.shape)
    flatM = K.reshape(M, (d1*d2*d3, d4))
    return K.sum(K.square(tf.norm(flatM, axis=1)))


def frobenius(M):
    d1, d2, d3, d4 = map(int, M.shape)
    entryPerW = K.reshape(M, (d1*d2, d3, d4))
    return K.sum(tf.norm(entryPerW, ord='fro', axis=(0,1)))


def operator_norm(M):
    d1, d2, d3, d4 = map(int, M.shape)
    entryPerW = K.reshape(M, (d1*d2, d3, d4))
    # ord 2 is operator norm: https://stackoverflow.com/questions/48275198/how-does-numpy-linalg-norm-ord-2-work
    return K.sum(tf.norm(entryPerW, ord=2)) 

T = tf.Variable(tf.ones([3,4,3,3]))
#print(T)
print(l2_row(T))
#print(frobenius(T))
#print(operator_norm(T))



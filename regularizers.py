from keras import backend as K
import tensorflow as tf

def combined_regularizer(regularizers, weights):
    if len(regularizers) == 0:
        return None

    assert len(regularizers) == len(weights)
    
    def reg_fn(W):
        loss = 0
        for i in range(len(regularizers)):
            loss += weights[i] * regularizers[i](W)
        return loss
    return reg_fn


def l21(W):
    d1, d2, d3, d4 = map(int, W.shape)
    all_W_rows = K.reshape(W, (d1*d2*d3, d4))
    return K.sum(tf.norm(all_W_rows, axis=1))


def operator_norm(W):
    d1, d2, d3, d4 = map(int, W.shape)
    all_W_rows = K.reshape(W, (d1*d2, d3, d4))
    # ord 2 is operator norm: https://stackoverflow.com/questions/48275198/how-does-numpy-linalg-norm-ord-2-work
    return K.sum(tf.norm(all_W_rows, ord=2))
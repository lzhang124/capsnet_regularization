from keras import backend as K
import tensorflow as tf


def weighted_regularizer(regularizer, weight):
    if regularizer is None:
        return None
    def reg_fn(W):
        return regularizer(W) * weight
    return reg_fn


def l21(W):
    d1 = list(map(int, W.shape))[-2]
    all_W_rows = K.reshape(W, (-1, d1))
    a = K.sum(tf.norm(all_W_rows, axis=1))
    print(a)
    return a


def operator_norm(W):
    d1, d2 = list(map(int, W.shape))[-2:]
    all_Ws = K.reshape(W, (-1, d1, d2))
    # ord 2 is operator norm: https://stackoverflow.com/questions/48275198/how-does-numpy-linalg-norm-ord-2-work
    return K.sum(tf.norm(all_Ws, ord=2))

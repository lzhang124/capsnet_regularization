from keras import backend as K
from keras import regularizers
import tensorflow as tf


def weighted_regularizer(regularizer, weight):
    if regularizer is None:
        return None
    def reg_fn(W):
        return weight * regularizer(W)
    return reg_fn


def l1(W):
    return regularizers.l1(1.)(W) / tf.size(W, out_type=tf.float32)


def l2(W):
    return regularizers.l1(2.)(W) / tf.size(W, out_type=tf.float32)


def l21(W):
    d1 = K.shape(W)[-2]
    all_W_rows = K.reshape(W, (-1, d1))
    return K.sum(tf.norm(all_W_rows, axis=1)) / tf.size(W, out_type=tf.float32)


def operator_norm(W):
    d1 = K.shape(W)[-2:]
    all_Ws = K.reshape(W, (-1, d1, d2))
    # ord 2 is operator norm: https://stackoverflow.com/questions/48275198/how-does-numpy-linalg-norm-ord-2-work
    return K.sum(tf.norm(all_Ws, ord=2)) / tf.size(W, out_type=tf.float32)

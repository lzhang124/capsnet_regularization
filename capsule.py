import keras.backend as K
import tensorflow as tf
from keras import initializers, layers
from keras.utils import conv_utils


def capsulize_fn(inputs):
    return K.expand_dims(inputs, axis=1)


def capsulize_output_shape(input_shape):
    return (input_shape[0],) + (1,) + input_shape[1:]


def length_fn(inputs):
    return K.sqrt(K.sum(K.square(inputs), -1))


def length_output_shape(input_shape):
    return input_shape[:-1]


def squash(vectors, axis=-1):
    '''
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    '''
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


def dim_transpose(order, n_dim, dim_i):
    new_order = []
    for i in order:
        if i < dim_i:
            new_order.append(i)
        elif i == dim_i:
            for n in range(n_dim):
                new_order.append(i + n)
        else:
            new_order.append(i + n_dim - 1)
    return tuple(new_order)


# class Mask(layers.Layer):
#     '
#     Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional 
#     input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
#     masked Tensor.
#     For example:
#         ```
#         x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
#         y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
#         out = Mask()(x)  # out.shape=[8, 6]
#         # or
#         out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
#         ```
#     '
#     def call(self, inputs, **kwargs):
#         if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
#             assert len(inputs) == 2
#             inputs, mask = inputs
#         else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
#             # compute lengths of capsules
#             x = K.sqrt(K.sum(K.square(inputs), -1))
#             # generate the mask which is a one-hot code.
#             # mask.shape=[None, n_classes]=[None, num_capsule]
#             mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

#         # inputs.shape=[None, num_capsule, dim_capsule]
#         # mask.shape=[None, num_capsule]
#         # masked.shape=[None, num_capsule * dim_capsule]
#         masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
#         return masked

#     def compute_output_shape(self, input_shape):
#         if type(input_shape[0]) is tuple:  # true label provided
#             return tuple([None, input_shape[0][1] * input_shape[0][2]])
#         else:  # no true label provided
#             return tuple([None, input_shape[1] * input_shape[2]])


class PrimaryCap:
    '''
    Apply Conv2D `num_capsules` times and concatenate all capsules
    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param kernel_size: dimension of each kernel
    :param strides: dimensions of strides
    :param padding: type of padding
    :param dilation_rate: dimensions of dilation
    '''
    def __init__(self, num_capsules, dim_capsule, kernel_size, strides=2, padding='valid', dilation_rate=1):
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')

    def __call__(self, inputs):
        output = layers.Conv2D(self.dim_capsule*self.num_capsules,
                               self.kernel_size,
                               strides=self.strides,
                               padding=self.padding,
                               dilation_rate=self.dilation_rate,
                               name='primarycap_conv2d')(inputs)
        outputs = layers.Reshape(target_shape=[-1, self.dim_capsule], name='primarycap_reshape')(output)
        return layers.Lambda(squash, name='primarycap_squash')(outputs)


class CapsuleLayer(layers.Layer):
    '''
    The capsule layer.
    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    '''
    def __init__(self, num_capsule, dim_capsule,
                 routings=3,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.out_num_capsule = num_capsule
        self.out_dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        assert len(input_shape) == 3, f'The input Tensor should have shape=(None, in_num_capsule, in_dim_capsule), got {input_shape}'
        self.in_num_capsule = input_shape[1]
        self.in_dim_capsule = input_shape[-1]

        # Transform matrix
        self.W = self.add_weight(shape=(self.in_num_capsule, self.out_num_capsule, self.in_dim_capsule, self.out_dim_capsule),
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape = (None, in_num_capsule, in_dim_capsule)
        # inputs_tiled.shape = (None, in_num_capsule, out_num_capsule, in_dim_capsule)
        inputs_tiled = K.repeat_elements(K.expand_dims(inputs, axis=2), self.out_num_capsule, 2)

        # inputs * W
        # x.shape = (in_num_capsule, out_num_capsule, in_dim_capsule)
        # W.shape = (in_num_capsule, out_num_capsule, in_dim_capsule, out_dim_capsule)
        # inputs_hat.shape = (None, in_num_capsule, out_num_capsule, out_dim_capsule)
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W), elems=inputs_tiled)

        # Routing algorithm -----------------------------------------------------------------------#
        # b.shape = (None, in_num_capsule, out_num_capsule)
        b = K.zeros(shape=(K.shape(inputs)[0], self.in_num_capsule, self.out_num_capsule))

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape = (None, in_num_capsule, out_num_capsule)
            c = K.softmax(b, axis=-1)

            # c.shape = (None, in_num_capsule, out_num_capsule)
            # inputs_hat.shape = (None, in_num_capsule, out_num_capsule, out_dim_capsule)
            # outputs.shape = (None, out_num_capsule, out_dim_capsule)
            outputs = squash(K.batch_dot(K.permute_dimensions(c, (0, 2, 1)), K.permute_dimensions(inputs_hat, (0, 2, 1, 3))), axis=-1)

            if i < self.routings - 1:
                # outputs.shape = (None, out_num_capsule, out_dim_capsule)
                # inputs_hat.shape = (None, in_num_capsule, out_num_capsule, out_dim_capsule)
                # b.shape = (None, in_num_capsule, out_num_capsule)
                b += K.permute_dimensions(K.batch_dot(outputs, K.permute_dimensions(inputs_hat, (0, 2, 3, 1))), (0, 2, 1))
        # -----------------------------------------------------------------------------------------#
        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.out_num_capsule, self.out_dim_capsule)

    def get_config(self):
        config = {
            'num_capsule': self.out_num_capsule,
            'dim_capsule': self.out_dim_capsule,
            'routings': self.routings
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvCapsuleLayer(layers.Layer):
    '''
    The convolutional capsule layer.
    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param kernel_size: dimension of each kernel
    :param strides: dimensions of strides
    :param padding: type of padding
    :param dilation_rate: dimensions of dilation
    :param routings: number of iterations for the routing algorithm
    '''
    def __init__(self, num_capsule, dim_capsule, kernel_size, strides=1, padding='valid', dilation_rate=1,
                 rank=2, transpose=False,
                 routings=3,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        self.transpose = transpose
        self.out_num_capsule = num_capsule
        self.out_dim_capsule = dim_capsule
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, self.rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, self.rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, self.rank, 'dilation_rate')
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        assert len(input_shape) >= 4, f'The input Tensor should have shape=(None, in_num_capsule, in_dim, in_dim_capsule), got {input_shape}'
        self.in_num_capsule = input_shape[1]
        self.in_dim_capsule = input_shape[-1]

        self.in_dim = input_shape[2:-1]
        self.out_dim = []
        for d in range(self.rank):
            dim = conv_utils.conv_output_length(
                self.in_dim[d],
                self.kernel_size[d],
                padding=self.padding,
                stride=self.strides[d],
                dilation=self.dilation_rate[d])
            self.out_dim.append(dim)
        self.out_dim = tuple(self.out_dim)

        # Transform matrix
        W_shape = (self.in_num_capsule, self.out_num_capsule) + self.kernel_size + (self.in_dim_capsule, self.out_dim_capsule)
        self.W = self.add_weight(shape=W_shape,
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 name='W')

        if self.transpose:
            self.conv_op = [K.conv1d_transpose, K.conv2d_transpose, K.conv3d_transpose][self.rank - 1]
        else:
            self.conv_op = [K.conv1d, K.conv2d, K.conv3d][self.rank - 1]

        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape = (None, in_num_capsule, in_dim, in_dim_capsule)
        # inputs_tiled.shape = (in_num_capsule, out_num_capsule, None, in_dim, in_dim_capsule)
        inputs_tiled = K.permute_dimensions(K.repeat_elements(K.expand_dims(inputs, axis=2), self.out_num_capsule, 2),
                                            dim_transpose((1, 2, 0, 3, 4), self.rank, 3))

        # inputs * W
        # inputs_tiled.shape = (in_num_capsule, out_num_capsule, None, in_dim, in_dim_capsule)
        # W.shape = (in_num_capsule, out_num_capsule, kernel_size, in_dim_capsule, out_dim_capsule)
        # inputs_hat.shape = (None, in_num_capsule, out_num_capsule, out_dim, out_dim_capsule)
        def conv_map(e):
            return K.map_fn(lambda (t, w): self.conv_op(t, w,
                                                        strides=self.strides,
                                                        padding=self.padding,
                                                        dilation_rate=self.dilation_rate),
                            elems=e, dtype=tf.float32)
        inputs_hat = K.permute_dimensions(K.map_fn(conv_map, elems=(inputs_tiled, self.W), dtype=tf.float32),
                                          dim_transpose((2, 0, 1, 3, 4), self.rank, 3))

        # Routing algorithm -----------------------------------------------------------------------#
        # b.shape = (None, in_num_capsule, out_num_capsule)
        b = K.zeros(shape=(K.shape(inputs)[0], self.in_num_capsule, self.out_num_capsule))

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape = (None, in_num_capsule, out_num_capsule, out_dim)
            c = K.softmax(b, axis=-1)
            for d in range(self.rank):
                c = K.repeat_elements(K.expand_dims(c, axis=-1), self.out_dim[d], -1)

            # c.shape = (None, in_num_capsule, out_num_capsule, out_dim)
            # inputs_hat.shape = (None, in_num_capsule, out_num_capsule, out_dim, out_dim_capsule)
            # outputs.shape = (None, out_num_capsule, out_dim, out_dim_capsule)
            outputs = squash(K.batch_dot(K.permute_dimensions(c, dim_transpose((0, 2, 3, 1), self.rank, 3)),
                                         K.permute_dimensions(inputs_hat, dim_transpose((0, 2, 3, 1, 4), self.rank, 3))), axis=-1)

            if i < self.routings - 1:
                # outputs.shape = (None, out_num_capsule, out_dim, out_dim_capsule)
                # inputs_hat.shape = (None, in_num_capsule, out_num_capsule, out_dim, out_dim_capsule
                # s.shape = (None, in_num_capsule, out_num_capsule, out_dim)
                # b.shape = (None, in_num_capsule, out_num_capsule)
                s = K.permute_dimensions(K.batch_dot(outputs, K.permute_dimensions(inputs_hat, dim_transpose((0, 2, 3, 4, 1), self.rank, 3))),
                                         dim_transpose((0, 3, 1, 2), self.rank, 2))
                for i in range(self.rank):
                    s = K.sum(s, axis=-1)
                b += s
        # -----------------------------------------------------------------------------------------#
        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.out_num_capsule) + self.out_dim + (self.out_dim_capsule,)

    def get_config(self):
        config = {
            'rank': self.rank,
            'transpose': self.transpose,
            'num_capsule': self.out_num_capsule,
            'dim_capsule': self.out_dim_capsule,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'routings': self.routings
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

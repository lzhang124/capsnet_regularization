import keras.backend as K
import tensorflow as tf
from keras import initializers, layers
from keras.utils import conv_utils


def capsulize_fn(inputs):
    return K.expand_dims(inputs, axis=1)


def length_fn(inputs):
    return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())


def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


def mask(inputs):
    '''
    Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional 
    input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros.
    For example:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
        out = Mask()(x)  # out.shape=[8, 3, 2]
        # or
        out2 = Mask()([x, y])  # out2.shape=[8, 3, 2]. Masked with true labels y. Of course y can also be manipulated.
        ```
    '''
    if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
        assert len(inputs) == 2
        inputs, mask = inputs
    else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
        # compute lengths of capsules
        x = K.sqrt(K.sum(K.square(inputs), -1))
        # generate the mask which is a one-hot code.
        # mask.shape=[None, n_classes]=[None, num_capsule]
        mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

    # inputs.shape=[None, num_capsule, dim_capsule]
    # mask.shape=[None, num_capsule]
    # masked.shape=[None, num_capsule, dim_capsule]
    masked = inputs * K.expand_dims(mask, -1)
    return masked


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


class PrimaryCap:
    '''
    Apply Conv2D `num_capsule` times and concatenate all capsules
    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param kernel_size: dimension of each kernel
    :param strides: dimensions of strides
    :param padding: type of padding
    :param dilation_rate: dimensions of dilation
    '''
    def __init__(self, num_capsule, dim_capsule, kernel_size, strides=2, padding='valid', dilation_rate=1):
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')

    def __call__(self, inputs):
        output = layers.Conv2D(self.dim_capsule*self.num_capsule,
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


class ConvCapsuleLayer:
    '''
    The convolutional capsule layer.
    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param kernel_size: dimension of each kernel
    :param strides: dimensions of strides
    :param padding: type of padding
    :param dilation_rate: dimensions of dilation
    :param transpose: if convolution operation should be a tranpose convolution
    :param routings: number of iterations for the routing algorithm
    '''
    def __init__(self, num_capsule, dim_capsule, kernel_size, strides=1, padding='valid', dilation_rate=1,
                 transpose=False, routings=3):
        self.out_num_capsule = num_capsule
        self.out_dim_capsule = dim_capsule
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.conv_layer = layers.Conv2DTranspose if transpose else layers.Conv2D
        self.routings = routings

    def route(self, inputs):
        # inputs.shape = (None, out_dim, in_num_capsule*out_num_capsule*out_dim_capsule)
        # inputs_hat.shape = (None, in_num_capsule, out_num_capsule, out_dim, out_dim_capsule)
        inputs_hat = K.reshape(inputs, (-1,) + K.int_shape(inputs)[1:3] + (self.in_num_capsule, self.out_num_capsule, self.out_dim_capsule))
        inputs_hat = K.permute_dimensions(inputs_hat, dim_transpose((0, 2, 3, 1, 4), 2, 1))

        # Routing algorithm -----------------------------------------------------------------------#
        # b.shape = (None, in_num_capsule, out_num_capsule)
        b = K.zeros(shape=(K.shape(inputs)[0], self.in_num_capsule, self.out_num_capsule))

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape = (None, in_num_capsule, out_num_capsule, out_dim)
            c = K.softmax(b, axis=-1)
            for d in range(2):
                c = K.repeat_elements(K.expand_dims(c, axis=-1), K.int_shape(inputs)[1:3][d], -1)

            # c.shape = (None, in_num_capsule, out_num_capsule, out_dim)
            # inputs_hat.shape = (None, in_num_capsule, out_num_capsule, out_dim, out_dim_capsule)
            # outputs.shape = (None, out_num_capsule, out_dim, out_dim_capsule)
            outputs = squash(K.batch_dot(K.permute_dimensions(c, dim_transpose((0, 2, 3, 1), 2, 3)),
                                         K.permute_dimensions(inputs_hat, dim_transpose((0, 2, 3, 1, 4), 2, 3))), axis=-1)

            if i < self.routings - 1:
                # outputs.shape = (None, out_num_capsule, out_dim, out_dim_capsule)
                # inputs_hat.shape = (None, in_num_capsule, out_num_capsule, out_dim, out_dim_capsule
                # s.shape = (None, in_num_capsule, out_num_capsule, out_dim)
                # b.shape = (None, in_num_capsule, out_num_capsule)
                s = K.permute_dimensions(K.batch_dot(outputs, K.permute_dimensions(inputs_hat, dim_transpose((0, 2, 3, 4, 1), 2, 3))),
                                         dim_transpose((0, 3, 1, 2), 2, 2))
                for i in range(2):
                    s = K.sum(s, axis=-1)
                b += s
        # -----------------------------------------------------------------------------------------#
        return outputs

    def __call__(self, inputs):
        _, self.in_num_capsule, _, _, self.in_dim_capsule = K.int_shape(inputs)

        capsules = []
        for i in range(self.in_num_capsule):
            in_cap = layers.Lambda(lambda x: x[:,i,...])(inputs)
            capsules.append(self.conv_layer(self.out_num_capsule*self.out_dim_capsule,
                                            kernel_size=self.kernel_size,
                                            strides=self.strides,
                                            padding=self.padding,
                                            dilation_rate=self.dilation_rate)(in_cap))
        if self.in_num_capsule > 1:
            outputs = layers.concatenate(capsules)
        else:
            outputs = capsules[0]

        return layers.Lambda(self.route)(outputs)

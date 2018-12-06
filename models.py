from keras import layers
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l1, l2
import capsule
import numpy as np
import os
import regularizers


REGULARIZER = {
    'l1': l1(1.),
    'l2': l2(1.),
    'l21': regularizers.l21,
    'operator_norm': regularizers.operator_norm,
}


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, axis=1))


class BaseModel:
    def __init__(self,
                 name,
                 n_class,
                 image_shape,
                 loss,
                 regularizers=None,
                 regularizer_weights=None,
                 save_freq=None,
                 tensorboard=None,
                 filename=None):
        self.name = name
        self.n_class = n_class
        self.image_shape = image_shape
        self.loss = loss
        self.regularizers = [REGULARIZER[reg] for reg in regularizers] if regularizers else []
        self.regularizer_weights = regularizer_weights if regularizer_weights else []
        self.save_freq = save_freq
        self.tensorboard = tensorboard

        self._new_model()
        if filename is not None:
            self.model.load_weights(filename)
        self._compile()

    def _new_model(self):
        raise NotImplementedError()

    def _compile(self):
        self.model.compile(optimizer=Adam(lr=1e-4), loss=self.loss, metrics=['accuracy'])

    def save(self):
        self.model.save(f'models/{self.name}.h5')

    def train(self, generator, val_gen, epochs):
        path = f'models/{self.name}/'
        os.makedirs(path, exist_ok=True)
        callbacks = []
        if self.save_freq:
            callbacks.append(ModelCheckpoint(path + '{epoch:0>3d}_{val_acc:.4f}.h5', save_weights_only=True, period=self.save_freq))
        if self.tensorboard:
            callbacks.append(TensorBoard(log_dir=f'./logs/{self.name}'))
        self.model.fit_generator(generator,
                                 epochs=epochs,
                                 validation_data=val_gen,
                                 verbose=1,
                                 callbacks=callbacks)

    def predict(self, generator):
        return self.model.predict_generator(generator, verbose=1)

    def test(self, generator):
        metrics = self.model.evaluate_generator(generator)
        return dict(zip(self.model.metrics_names, [metrics] if isinstance(metrics, float) else metrics))


class ConvNet(BaseModel):
    def _new_model(self):
        inputs = layers.Input(shape=self.image_shape)

        conv1 = layers.Conv2D(4, 3, activation='relu', padding='same')(inputs)
        pool1 = layers.MaxPooling2D(2)(conv1)

        conv2 = layers.Conv2D(8, 3, activation='relu', padding='same')(pool1)
        pool2 = layers.MaxPooling2D(2)(conv2)

        conv3 = layers.Conv2D(16, 3, activation='relu', padding='same')(pool2)
        pool3 = layers.MaxPooling2D(2)(conv3)

        flat = layers.Flatten()(pool3)
        drop = layers.Dropout(0.25)(flat)
        fc = layers.Dense(32, activation='relu')(drop)

        outputs = layers.Dense(self.n_class, activation='sigmoid')(fc)

        self.model = Model(inputs=inputs, outputs=outputs)


class Autoencoder(BaseModel):
    def _new_model(self):
        inputs = layers.Input(shape=self.image_shape)

        conv1 = layers.Conv2D(4, 3, activation='relu', padding='same')(inputs)
        pool1 = layers.MaxPooling2D(2)(conv1)

        conv2 = layers.Conv2D(4, 3, activation='relu', padding='same')(pool1)
        pool2 = layers.MaxPooling2D(2)(conv2)

        conv3 = layers.Conv2D(4, 3, activation='relu', padding='same')(pool2)
        pool3 = layers.MaxPooling2D(2)(conv3)

        conv4 = layers.Conv2D(4, 3, activation='relu', padding='same')(pool3)

        up5 = layers.Conv2DTranspose(4, 2, strides=2, padding='same')(conv4)
        conv5 = layers.Conv2D(4, 3, activation='relu', padding='same')(up5)

        up6 = layers.Conv2DTranspose(4, 2, strides=2, padding='same')(conv5)
        conv6 = layers.Conv2D(4, 3, activation='relu', padding='same')(up6)

        up7 = layers.Conv2DTranspose(4, 2, strides=2, padding='same')(conv6)
        conv7 = layers.Conv2D(4, 3, activation='relu', padding='same')(up7)

        outputs = layers.Conv2D(self.image_shape[-1], (1, 1), activation='sigmoid')(conv7)

        self.model = Model(inputs=inputs, outputs=outputs)

    def _compile(self):
        self.model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])


class CapsNet(BaseModel):
    def _new_model(self):
        inputs = layers.Input(shape=self.image_shape)

        # Layer 1: Just a conventional Conv2D layer
        conv1 = layers.Conv2D(256, 9, padding='valid', activation='relu')(inputs)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
        primarycaps = capsule.PrimaryCap(conv1, num_capsules=32, dim_capsule=8, kernel_size=9, strides=2, padding='valid')

        # Layer 3: Capsule layer. Routing algorithm works here.
        digitcaps = capsule.CapsuleLayer(self.n_class,
                                         dim_capsule=16,
                                         kernel_regularizer=regularizers.combined_regularizer(self.regularizers, self.regularizer_weights),
                                         name='digitcaps')(primarycaps)

        # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
        outputs = layers.Lambda(capsule.length_fn, capsule.length_output_shape, name='length')(digitcaps)

        self.model = Model(inputs=inputs, outputs=outputs)


class ConvCapsNet(BaseModel):
    def _new_model(self):
        inputs = layers.Input(shape=self.image_shape)

        conv1 = layers.Conv2D(256, kernel_size=9, padding='valid', activation='relu')(inputs)
        conv1 = K.expand_dims(conv1, axis=1)
        
        convcaps = capsule.ConvCapsuleLayer(32, dim_capsule=8, kernel_size=9, strides=2, padding='valid')(conv1)
        flat = layers.Reshape((-1, 8))(convcaps)
        
        digitcaps = capsule.CapsuleLayer(self.n_class, dim_capsule=16, name='digitcaps')(flat)
        
        outputs = layers.Lambda(capsule.length_fn, capsule.length_output_shape, name='length')(digitcaps)

        self.model = Model(inputs=inputs, outputs=outputs)


class FullCaps(BaseModel):
    def _new_model(self):
        inputs = layers.Input(shape=self.image_shape)
        inputs_reshape = K.expand_dims(inputs, axis=1)

        convcaps1 = capsule.ConvCapsuleLayer(64, dim_capsule=4, kernel_size=9, padding='valid')(inputs_reshape)
        convcaps2 = capsule.ConvCapsuleLayer(32, dim_capsule=8, kernel_size=9, strides=2, padding='valid')(convcaps1)
        flat = layers.Reshape((-1, 8))(convcaps2)

        digitcaps = capsule.CapsuleLayer(self.n_class, dim_capsule=16, name='digitcaps')(flat)
        
        outputs = layers.Lambda(capsule.length_fn, capsule.length_output_shape, name='length')(digitcaps)

        self.model = Model(inputs=inputs, outputs=outputs)

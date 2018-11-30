from keras import layers
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import capsule
import numpy as np


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


class BaseModel:
    def __init__(self, name, n_class, image_shape, loss, tensorboard, routings=None, filename=None):
        self.name = name
        self.n_class = n_class
        self.image_shape = image_shape
        self.loss = loss
        self.tensorboard = tensorboard
        self.routings = routings

        self._new_model()
        if filename is not None:
            self.model.load_weights(filename)
        self._compile()

    def _new_model(self):
        raise NotImplementedError()

    def _compile(self):
        raise NotImplementedError()

    def save(self):
        self.model.save(f'models/{self.name}.h5')

    def train(self, generator, val_gen, epochs):
        callbacks = [TensorBoard(log_dir=f'./logs/{self.name}')] if self.tensorboard else []
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

        conv1 = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(inputs)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(pool1)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(pool2)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        flat = layers.Flatten()(pool3)
        drop = layers.Dropout(0.25)(flat)
        fc = layers.Dense(32, activation='relu')(drop)

        outputs = layers.Dense(self.n_class, activation='sigmoid')(fc)

        self.model = Model(inputs=inputs, outputs=outputs)

    def _compile(self):
        self.model.compile(optimizer=Adam(lr=1e-4), loss=self.loss)


class Autoencoder(BaseModel):
    def _new_model(self):
        inputs = layers.Input(shape=self.image_shape)

        conv1 = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(inputs)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(pool1)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(pool2)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(pool3)

        up5 = layers.Conv2DTranspose(4, (2, 2), strides=(2, 2), padding='same')(conv4)
        conv5 = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(up5)

        up6 = layers.Conv2DTranspose(4, (2, 2), strides=(2, 2), padding='same')(conv5)
        conv6 = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(up6)

        up7 = layers.Conv2DTranspose(4, (2, 2), strides=(2, 2), padding='same')(conv6)
        conv7 = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(up7)

        outputs = layers.Conv2D(self.image_shape[-1], (1, 1), activation='sigmoid')(conv7)

        self.model = Model(inputs=inputs, outputs=outputs)

    def _compile(self):
        self.model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])


class CapsNet(BaseModel):
    def _new_model(self):
        inputs = layers.Input(shape=self.image_shape)

        # Layer 1: Just a conventional Conv2D layer
        conv1 = layers.Conv2D(256, (9, 9), padding='valid', activation='relu')(inputs)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
        primarycaps = capsule.PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

        # Layer 3: Capsule layer. Routing algorithm works here.
        digitcaps = capsule.CapsuleLayer(num_capsule=self.n_class, dim_capsule=16, routings=self.routings, name='digitcaps')(primarycaps)

        # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
        outputs = layers.Lambda(capsule.length, capsule.length_output_shape)(digitcaps)

        self.model = Model(inputs=inputs, outputs=outputs)

    def _compile(self):
        self.model.compile(optimizer=Adam(lr=0.001), loss=self.loss)

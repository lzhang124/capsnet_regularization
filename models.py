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


class BaseModel:
    def __init__(self,
                 name,
                 n_class,
                 image_shape,
                 regularizer=None,
                 regularizer_weight=None,
                 lr=1e-4,
                 metrics=None,
                 save_freq=None,
                 tensorboard=None,
                 filename=None):
        self.name = name
        self.n_class = n_class
        self.image_shape = image_shape
        self.regularizer = REGULARIZER[regularizer] if regularizer else None
        self.regularizer_weight = regularizer_weight if regularizer_weight else 0
        self.lr = lr
        self.metrics = ['accuracy'] if metrics is None else metrics
        self.save_freq = save_freq
        self.tensorboard = tensorboard

        self._new_model()
        if filename is not None:
            self.model.load_weights(filename)
        self._compile()

    def _new_model(self):
        raise NotImplementedError()

    def _compile(self):
        self.model.compile(optimizer=Adam(lr=self.lr), loss='categorical_crossentropy', metrics=self.metrics)

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

        conv1 = layers.Conv2D(256, 9, activation='relu', padding='valid')(inputs)
        pool1 = layers.MaxPooling2D(2)(conv1)

        conv2 = layers.Conv2D(256, 9, strides=2, activation='relu', padding='valid')(pool1)
        pool2 = layers.MaxPooling2D(2)(conv2)

        flat = layers.Flatten()(pool3)
        drop = layers.Dropout(0.25)(flat)
        fc = layers.Dense(128, activation='relu')(drop)

        outputs = layers.Dense(self.n_class, activation='sigmoid')(fc)

        self.model = Model(inputs=inputs, outputs=outputs)


class Autoencoder(BaseModel):
    def _new_model(self):
        inputs = layers.Input(shape=self.image_shape)

        conv1 = layers.Conv2D(4, 3, activation='relu', padding='valid')(inputs)
        pool1 = layers.MaxPooling2D(2)(conv1)

        conv2 = layers.Conv2D(8, 3, activation='relu', padding='valid')(pool1)
        pool2 = layers.MaxPooling2D(2)(conv2)

        conv3 = layers.Conv2D(16, 3, activation='relu', padding='valid')(pool2)
        pool3 = layers.MaxPooling2D(2)(conv3)

        conv4 = layers.Conv2D(32, 3, activation='relu', padding='valid')(pool3)

        up5 = layers.Conv2DTranspose(4, 2, strides=2, padding='valid')(conv4)
        conv5 = layers.Conv2D(16, 3, activation='relu', padding='valid')(up5)

        up6 = layers.Conv2DTranspose(4, 2, strides=2, padding='valid')(conv5)
        conv6 = layers.Conv2D(8, 3, activation='relu', padding='valid')(up6)

        up7 = layers.Conv2DTranspose(4, 2, strides=2, padding='valid')(conv6)
        conv7 = layers.Conv2D(4, 3, activation='relu', padding='valid')(up7)

        outputs = layers.Conv2D(self.image_shape[-1], (1, 1), activation='sigmoid')(conv7)

        self.model = Model(inputs=inputs, outputs=outputs)

    def _compile(self):
        self.model.compile(optimizer=Adam(lr=self.lr), loss='mse', metrics=self.metrics)


class CapsNet(BaseModel):
    def _new_model(self):
        inputs = layers.Input(shape=self.image_shape)

        conv1 = layers.Conv2D(256, 9, padding='valid', activation='relu')(inputs)

        primarycaps = capsule.PrimaryCap(32, 8, 9, strides=2, padding='valid')(conv1)

        digitcaps = capsule.CapsuleLayer(self.n_class, 16,
                                         kernel_regularizer=regularizers.weighted_regularizer(self.regularizer, self.regularizer_weight),
                                         name='digitcaps')(primarycaps)

        outputs = layers.Lambda(capsule.length_fn, capsule.length_output_shape, name='length')(digitcaps)

        self.model = Model(inputs=inputs, outputs=outputs)


class ConvCaps(BaseModel):
    def _new_model(self):
        inputs = layers.Input(shape=self.image_shape)

        conv1 = layers.Conv2D(256, 9, padding='valid', activation='relu')(inputs)
        conv1 = layers.Lambda(capsule.capsulize_fn, capsule.capsulize_output_shape, name='capsulize')(conv1)
        
        convcaps = capsule.ConvCapsuleLayer(32, 8, 9, strides=2, padding='valid')(conv1)
        flat = layers.Reshape((-1, 8))(convcaps)
        
        digitcaps = capsule.CapsuleLayer(self.n_class, 16, name='digitcaps')(flat)
        
        outputs = layers.Lambda(capsule.length_fn, capsule.length_output_shape, name='length')(digitcaps)

        self.model = Model(inputs=inputs, outputs=outputs)


class FullCaps(BaseModel):
    def _new_model(self):
        inputs = layers.Input(shape=self.image_shape)
        inputs_reshape = layers.Lambda(capsule.capsulize_fn, capsule.capsulize_output_shape, name='capsulize')(inputs)

        convcaps1 = capsule.ConvCapsuleLayer(64, 4, 9, padding='valid')(inputs_reshape)
        convcaps2 = capsule.ConvCapsuleLayer(32, 8, 9, strides=2, padding='valid')(convcaps1)
        flat = layers.Reshape((-1, 8))(convcaps2)

        digitcaps = capsule.CapsuleLayer(self.n_class, 16, name='digitcaps')(flat)
        
        outputs = layers.Lambda(capsule.length_fn, capsule.length_output_shape, name='length')(digitcaps)

        self.model = Model(inputs=inputs, outputs=outputs)

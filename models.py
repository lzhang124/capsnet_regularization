import numpy as np
import os
import util
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.callbacks import TensorBoard
from keras import backend as K
from keras import layers
from process import uncrop


class BaseModel:
    def __init__(self, input_size, name=None, filename=None):
        self.input_size = input_size
        self.name = name if name else self.__class__.__name__.lower()
        self._new_model()
        if filename is not None:
            self.model.load_weights(filename)

    def _new_model(self):
        raise NotImplementedError()        

    def save(self):
        self.model.save('models/{}.h5'.format(self.name))

    def compile(self, weight=None, loss=None):
        raise NotImplementedError()

    def train(self, generator, val_gen, epochs):
        self.model.fit_generator(generator,
                                 epochs=epochs,
                                 validation_data=val_gen,
                                 verbose=1,
                                 callbacks=[TensorBoard(log_dir='./logs/{}'.format(self.name))])

    def predict(self, generator, path):
        preds = self.model.predict_generator(generator, verbose=1)
        save_predictions(preds, generator, path)

    def test(self, generator):
        return self.model.evaluate_generator(generator)


class UNet(BaseModel):
    # 140 perceptive field
    def _new_model(self):
        inputs = layers.Input(shape=self.input_size)

        conv1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
        conv1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)

        conv2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
        conv2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)

        conv3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
        conv3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)

        conv4 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
        conv4 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
        pool4 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv4)

        conv5 = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
        conv5 = layers.Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)

        up6 = layers.Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5)
        conc6 = layers.concatenate([up6, conv4])
        conv6 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conc6)
        conv6 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)

        up7 = layers.Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6)
        conc7 = layers.concatenate([up7, conv3])
        conv7 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conc7)
        conv7 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)

        up8 = layers.Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7)
        conc8 = layers.concatenate([up8, conv2])
        conv8 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conc8)
        conv8 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)

        up9 = layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8)
        conc9 = layers.concatenate([up9, conv1])
        conv9 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conc9)
        conv9 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)

        outputs = layers.Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)

        self.model = Model(inputs=inputs, outputs=outputs)

    def compile(self, weight=None, loss=None):
        self.model.compile(optimizer=Adam(lr=1e-4),
                           loss=weighted_crossentropy(weight=weight, boundary_weight=loss),
                           metrics=[dice_coef])
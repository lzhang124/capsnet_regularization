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


class ConvNet(BaseModel):
    def _new_model(self):
        inputs = layers.Input(shape=self.input_size)

        conv1 = layers.Conv2D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
        conv1 = layers.Conv2D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2, 2))(conv1)

        conv2 = layers.Conv2D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2, 2))(conv2)

        conv3 = layers.Conv2D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2, 2))(conv3)

        flat4 = layers.Flatten()(pool3)
        fc5 = layers.Dense(1024, activation='relu')(flat4)
        fc6 = layers.Dense(1024, activation='relu')(fc5)

        outputs = layers.Dense(3, activation='sigmoid')(fc6)

        self.model = Model(inputs=inputs, outputs=outputs)

    def compile(self, weight=None, loss=None):
        self.model.compile(optimizer=Adam(lr=1e-4),
                           loss='mean_squared_error',
                           metrics=['accuracy'])
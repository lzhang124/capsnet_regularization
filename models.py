from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras import layers


class BaseModel:
    def __init__(self, name, image_shape, tensorboard, filename=None):
        self.name = name
        self.image_shape = image_shape
        self.tensorboard = tensorboard

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
        return self.model.evaluate_generator(generator)


class ConvNet(BaseModel):
    def _new_model(self):
        inputs = layers.Input(shape=self.image_shape)

        conv1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        flat3 = layers.Flatten()(pool2)
        fc3 = layers.Dense(32, activation='relu')(flat3)

        outputs = layers.Dense(3, activation='sigmoid')(fc3)

        self.model = Model(inputs=inputs, outputs=outputs)

    def _compile(self):
        self.model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error')


class Autoencoder(BaseModel):
    def _new_model(self):
        inputs = layers.Input(shape=self.image_shape)

        conv1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)

        up4 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv3)
        conv4 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(up4)
        conv4 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(conv4)

        up5 = layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(conv4)
        conv5 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(up5)
        conv5 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(conv5)

        outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(conv5)

        self.model = Model(inputs=inputs, outputs=outputs)

    def _compile(self):
        self.model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics='accuracy')


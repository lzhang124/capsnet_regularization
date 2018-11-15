from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras import layers


class BaseModel:
    def __init__(self, name, image_shape=(64, 64, 3), filename=None):
        self.image_shape = image_shape
        self.name = name
        self._new_model()
        if filename is not None:
            self.model.load_weights(filename)

    def _new_model(self):
        raise NotImplementedError()

    def save(self):
        self.model.save('models/{}.h5'.format(self.name))

    def compile(self):
        raise NotImplementedError()

    def train(self, generator, val_gen, epochs):
        self.model.fit_generator(generator,
                                 epochs=epochs,
                                 validation_data=val_gen,
                                 verbose=1,
                                 callbacks=[TensorBoard(log_dir='./logs/{}'.format(self.name))])

    def predict(self, generator):
        preds = self.model.predict_generator(generator, verbose=1)
        return preds

    def test(self, generator):
        return self.model.evaluate_generator(generator)


class ConvNet(BaseModel):
    def _new_model(self):
        inputs = layers.Input(shape=self.image_shape)

        conv1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(pool1)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        flat3 = layers.Flatten()(pool2)
        fc3 = layers.Dense(32, activation='relu')(flat3)

        outputs = layers.Dense(3, activation='sigmoid')(fc3)

        self.model = Model(inputs=inputs, outputs=outputs)

    def compile(self):
        self.model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=['accuracy'])


class Autoencoder(BaseModel):
    def _new_model(self):
        inputs = layers.Input(shape=self.image_shape)

        conv1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(pool1)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)

        up4 = layers.Conv2DTranspose(16, (2, 2))(conv3)
        conv4 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(up4)

        up5 = layers.Conv2DTranspose(8, (2, 2))(conv4)
        conv5 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(up5)

        outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(conv5)

        self.model = Model(inputs=inputs, outputs=outputs)

    def compile(self):
        self.model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])


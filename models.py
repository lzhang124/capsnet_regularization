from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras import layers


class BaseModel:
    def __init__(self, name, image_shape=(256, 256, 3), filename=None):
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

        conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        flat5 = layers.Flatten()(pool4)
        fc6 = layers.Dense(1024, activation='relu')(flat5)
        fc6 = layers.Dense(1024, activation='relu')(fc6)

        outputs = layers.Dense(3, activation='sigmoid')(fc6)

        self.model = Model(inputs=inputs, outputs=outputs)

    def compile(self):
        self.model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=['accuracy'])


class Autoencoder(BaseModel):
    def _new_model(self):
        inputs = layers.Input(shape=self.image_shape)

        conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

        up6 = layers.Conv2DTranspose(512, (2, 2))(conv5)
        conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
        conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

        up7 = layers.Conv2DTranspose(256, (2, 2))(conv6)
        conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
        conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

        up8 = layers.Conv2DTranspose(128, (2, 2))(conv7)
        conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
        conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

        up9 = layers.Conv2DTranspose(64, (2, 2))(conv8)
        conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
        conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        self.model = Model(inputs=inputs, outputs=outputs)

    def compile(self):
        self.model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])


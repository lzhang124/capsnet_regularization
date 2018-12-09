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


class Decoder:
    def __init__(self, image_shape, mask, conv):
        self.image_shape = image_shape
        self.mask = mask
        self.conv = conv

    def __call__(self, inputs):
        if self.mask:
            masked = layers.Lambda(capsule.mask)(inputs)
        else:   
            masked = layers.Flatten()(inputs)

        if self.conv:
            fc = layers.Dense(256, activation='relu')(masked)
            reshape = layers.Reshape((8, 8, 4))(fc)
            up1 = layers.Conv2DTranspose(256, 9, strides=2, activation='relu', padding='valid')(reshape)
            up2 = layers.Conv2DTranspose(256, 9, activation='relu', padding='valid')(up1)
            outputs = layers.Conv2D(self.image_shape[-1], 1, activation='sigmoid')(up2)
        else:
            up1 = layers.Dense(512, activation='relu')(masked)
            up2 = layers.Dense(1024, activation='relu')(up1)
            outputs = layers.Dense(np.prod(self.image_shape), activation='sigmoid')(up2)
            outputs = layers.Reshape(self.image_shape, name='reconstruction')(outputs)
        return outputs


class BaseModel:
    def __init__(self,
                 name,
                 n_class,
                 image_shape,
                 lr,
                 decoder=False,
                 mask=False,
                 conv=False,
                 regularizer=None,
                 regularizer_weight=None,
                 save_freq=None,
                 tensorboard=None,
                 filename=None):
        self.name = name
        self.n_class = n_class
        self.image_shape = image_shape
        self.lr = lr
        self.decoder = Decoder(image_shape, mask, conv) if decoder else None
        self.mask = mask
        self.regularizer = REGULARIZER[regularizer] if regularizer is not None else None
        self.regularizer_weight = regularizer_weight
        self.save_freq = save_freq
        self.tensorboard = tensorboard

        self._new_model()
        if filename is not None:
            self.model.load_weights(filename)
        self._compile()

    def _new_model(self):
        raise NotImplementedError()

    def _compile(self):
        loss = ['categorical_crossentropy', 'mse'] if self.decoder else 'categorical_crossentropy'
        self.model.compile(optimizer=Adam(lr=self.lr), loss=loss, metrics=['accuracy'])

    def save(self):
        self.model.save(f'models/{self.name}.h5')

    def train(self, generator, val_gen, epochs):
        callbacks = []
        if self.save_freq:
            path = f'models/{self.name}/'
            os.makedirs(path, exist_ok=True)
            callbacks.append(ModelCheckpoint(path + '{epoch:0>3d}.h5', save_weights_only=True, period=self.save_freq))
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

        conv2 = layers.Conv2D(256, 9, strides=2, activation='relu', padding='valid')(conv1)

        flat = layers.Flatten()(conv2)
        fc = layers.Dense(128, activation='relu')(flat)

        outputs = layers.Dense(self.n_class, activation='sigmoid', name='classification')(fc)

        if self.decoder:
            y = layers.Input(shape=(self.n_class,))
            if self.mask:
                recon = self.decoder([outputs, y])
            else:
                recon = self.decoder(outputs)
            self.model = Model(inputs=[inputs, y], outputs=[outputs, recon])
        else:
            self.model = Model(inputs=inputs, outputs=outputs)


class CapsNet(BaseModel):
    def _new_model(self):
        inputs = layers.Input(shape=self.image_shape)

        conv1 = layers.Conv2D(256, 9, padding='valid', activation='relu')(inputs)

        primarycaps = capsule.PrimaryCap(32, 8, 9, strides=2, padding='valid')(conv1)

        digitcaps = capsule.CapsuleLayer(self.n_class, 16,
                                         kernel_regularizer=regularizers.weighted_regularizer(self.regularizer, self.regularizer_weight),
                                         name='digitcaps')(primarycaps)

        outputs = layers.Lambda(capsule.length_fn, name='length')(digitcaps)

        if self.decoder:
            y = layers.Input(shape=(self.n_class,))
            if self.mask:
                recon = self.decoder([digitcaps, y])
            else:
                recon = self.decoder(digitcaps)
            self.model = Model(inputs=[inputs, y], outputs=[outputs, recon])
        else:
            self.model = Model(inputs=inputs, outputs=outputs)


class ConvCaps(BaseModel):
    def _new_model(self):
        inputs = layers.Input(shape=self.image_shape)

        conv1 = layers.Conv2D(256, 9, padding='valid', activation='relu')(inputs)
        conv1 = layers.Lambda(capsule.capsulize_fn, name='capsulize')(conv1)
        
        convcaps = capsule.ConvCapsuleLayer(32, 8, 9, strides=2, padding='valid')(conv1)
        flat = layers.Reshape((-1, 8))(convcaps)
        
        digitcaps = capsule.CapsuleLayer(self.n_class, 16, name='digitcaps')(flat)
        
        outputs = layers.Lambda(capsule.length_fn, name='length')(digitcaps)

        self.model = Model(inputs=inputs, outputs=outputs)


class FullCaps(BaseModel):
    def _new_model(self):
        inputs = layers.Input(shape=self.image_shape)
        inputs_reshape = layers.Lambda(capsule.capsulize_fn, name='capsulize')(inputs)

        convcaps1 = capsule.ConvCapsuleLayer(64, 4, 9, padding='valid')(inputs_reshape)
        convcaps2 = capsule.ConvCapsuleLayer(32, 8, 9, strides=2, padding='valid')(convcaps1)
        flat = layers.Reshape((-1, 8))(convcaps2)

        digitcaps = capsule.CapsuleLayer(self.n_class, 16, name='digitcaps')(flat)
        
        outputs = layers.Lambda(capsule.length_fn, name='length')(digitcaps)

        self.model = Model(inputs=inputs, outputs=outputs)

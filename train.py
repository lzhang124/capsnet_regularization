import logging
logging.basicConfig(level=logging.INFO)

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--model',
                    help='Type of model',
                    dest='model', type=str, required=True)
parser.add_argument('--data',
                    help='Which dataset',
                    dest='data', type=str, required=True)
parser.add_argument('--name',
                    help='Name of model',
                    dest='name', type=str, required=True)
parser.add_argument('--regularizer',
                    help='Regularizer to use',
                    dest='regularizer', type=str)
parser.add_argument('--regularizer-weight',
                    help='Weight corresponding to regularizer',
                    dest='regularizer_weight', type=float)
parser.add_argument('--epochs',
                    help='Training epochs',
                    dest='epochs', type=int, required=True)
parser.add_argument('--batch-size',
                    help='Batch size',
                    dest='batch_size', type=int, default=1)
parser.add_argument('--save-freq',
                    help='Frequency of saving models',
                    dest='save_freq', type=int)
parser.add_argument('--tensorboard',
                    help='Enable tensorboard',
                    dest='tensorboard', action='store_true')
parser.add_argument('--model-file',
                    help='Pretrained model file',
                    dest='model_file', type=str)

options = parser.parse_args()

import os
import data
import datetime
import models
import numpy as np
import time
import util


MODELS = {
    'conv': models.ConvNet,
    'ae': models.Autoencoder,
    'caps': models.CapsNet,
    'convcaps': models.ConvCaps,
    'fullcaps': models.FullCaps,
}

DATA_GEN = {
    'mnist': data.MNISTGenerator,
    'cifar': data.CIFARGenerator,
}

CLASSES = {
    'mnist': 10,
    'cifar': 10,
}

IMAGE_SHAPE = {
    'mnist': (32, 32, 1),
    'cifar': (32, 32, 3),
}

LOSS = {
    'cifar': 'mse',
    'mnist': 'categorical_crossentropy',
}


def main(options):
    start = time.time()

    assert options.model in MODELS

    logging.info('Creating data generators.')
    data_gen = DATA_GEN[options.data]
    label_type = 'input' if options.model == 'ae' else 'label'
    train_gen = data_gen('train', batch_size=options.batch_size, label_type=label_type)
    val_gen = data_gen('val', batch_size=options.batch_size, label_type=label_type)
    pred_gen = data_gen('test', batch_size=1, shuffle=True)
    test_gen = data_gen('test', batch_size=1, label_type=label_type, shuffle=False)

    logging.info('Creating model.')
    m = MODELS[options.model](options.name,
                              CLASSES[options.data],
                              IMAGE_SHAPE[options.data],
                              regularizers=options.regularizer,
                              regularizer_weight=options.regularizer_weight,
                              save_freq=options.save_freq,
                              tensorboard=options.tensorboard,
                              filename=options.model_file)

    logging.info('Training model.')
    m.train(train_gen, val_gen, options.epochs)

    logging.info('Making predictions.')
    preds = m.predict(pred_gen)
    preds = np.argmax(preds[:20,...], axis=-1)
    logging.info(preds)
    # os.makedirs(f'data/{options.name}/', exist_ok=True)
    # for i in range(preds.shape[0]):
    #     util.save_img(pred_gen[i][0], f'data/{options.name}/{str(i).zfill(4)}_true.png')
    #     if options.model == 'ae':
    #         util.save_img(preds[i], f'data/{options.name}/{str(i).zfill(4)}.png')

    logging.info('Testing model.')
    metrics = m.test(test_gen)
    logging.info(metrics)

    end = time.time()
    logging.info(f'total time: {datetime.timedelta(seconds=(end - start))}')


if __name__ == '__main__':
    main(options)

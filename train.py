import logging
logging.basicConfig(level=logging.INFO)

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--model',
                    metavar='MODEL',
                    help='Type of model',
                    dest='model', type=str, required=True)
parser.add_argument('--data',
                    metavar='DATASET',
                    help='Which dataset',
                    dest='data', type=str, required=True)
parser.add_argument('--epochs',
                    metavar='EPOCHS',
                    help='Training epochs',
                    dest='epochs', type=int, required=True)
parser.add_argument('--name',
                    metavar='MODEL_NAME',
                    help='Name of model',
                    dest='name', type=str, required=True)
parser.add_argument('--tensorboard',
                    metavar='TENSORBOARD',
                    help='Enable tensorboard',
                    dest='tensorboard', action='store_true')
options = parser.parse_args()

import os
import cubes
import mnist
import models
import util


MODELS = {
    'conv': models.ConvNet,
    'ae': models.Autoencoder,
}

DATA_GEN = {
    'cubes': cubes.CubeGenerator,
    'mnist': mnist.MNISTGenerator,
}

LABEL = {
    'conv': {
        'cubes': 'pose',
        'mnist': 'digit',
    },
    'ae': {
        'cubes': 'input',
        'mnist': 'input',
    },
}

IMAGE_SHAPE = {
    'cubes': (32, 32, 3),
    'mnist': (32, 32, 1),
}


def main(options):
    logging.info('Creating data generators.')
    data_gen = DATA_GEN[options.data]
    label_type = LABEL[options.model][options.data]
    train_gen = data_gen(10, label_type=label_type)
    val_gen = data_gen(10, label_type=label_type)
    pred_gen = data_gen(10, shuffle=False)
    test_gen = data_gen(10, label_type=label_type, shuffle=False)

    logging.info('Creating model.')
    m = MODELS[options.model](options.name, IMAGE_SHAPE[options.data])

    logging.info('Training model.')
    m.train(train_gen, val_gen, options.epochs)

    if options.model == 'ae':
        logging.info('Making predictions.')
        preds = m.predict(pred_gen)
        for i in range(10):
            util.save_img(pred_gen[i][0], f'data/{i}_true.png')
            util.save_img(preds[i], f'data/{i}.png')

    logging.info('Testing model.')
    metrics = m.test(test_gen)
    logging.info(metrics)


if __name__ == '__main__':
    main(options)
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


DATASET = {
    'cubes': cubes.CubeGenerator,
    'mnist': mnist.MNISTGenerator,
}


def main(options):
    logging.info('Creating data generators.')
    train_gen = cubes.CubeGenerator(10, label_type='input')
    val_gen = cubes.CubeGenerator(10, label_type='input')
    pred_gen = cubes.CubeGenerator(10, shuffle=False)
    test_gen = cubes.CubeGenerator(10, label_type='input', shuffle=False)

    logging.info('Creating model.')
    m = MODELS[options.model](options.name)

    logging.info('Training model.')
    m.train(train_gen, val_gen, options.epochs)

    logging.info('Making predictions.')
    preds = m.predict(pred_gen)
    for i in range(10):
        util.save_img(pred_gen[i][0], 'data/{}_true.png'.format(i))
        util.save_img(preds[i], 'data/{}.png'.format(i))

    logging.info('Testing model.')
    metrics = m.test(test_gen)
    logging.info(metrics)


if __name__ == '__main__':
    main(options)
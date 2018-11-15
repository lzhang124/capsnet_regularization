import logging
logging.basicConfig(level=logging.INFO)

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--epochs',
                    metavar='EPOCHS',
                    help='Training epochs',
                    dest='epochs', type=int, default=1000)
parser.add_argument('--name',
                    metavar='MODEL_NAME',
                    help='Name of model',
                    dest='name', type=str, required=True)
options = parser.parse_args()

import os
import cubes
import models
import util


def main(options):
    train_gen = cubes.CubeGenerator(10, label_type='input')
    val_gen = cubes.CubeGenerator(10, label_type='input')
    pred_gen = cubes.CubeGenerator(10)
    test_gen = cubes.CubeGenerator(10, label_type='input')

    m = models.Autoencoder(options.name)
    m.compile()
    m.train(train_gen, val_gen, options.epochs)
    preds = m.predict(pred_gen)
    for i in range(10):
        util.save_img(preds[i], 'data/{}.png'.format(i))


if __name__ == '__main__':
    main(options)
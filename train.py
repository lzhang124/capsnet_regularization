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


def main(options):
    train_gen = cubes.CubeGenerator(10, label_type='pose')
    val_gen = cubes.CubeGenerator(10, label_type='pose')
    pred_gen = cubes.CubeGenerator(10)
    test_gen = cubes.CubeGenerator(10, label_type='pose')

    m = models.ConvNet(options.name)
    m.compile()
    m.train(train_gen, val_gen, options.epochs)
    preds = m.predict(pred_gen)
    print(preds)


if __name__ == '__main__':
    main(options)
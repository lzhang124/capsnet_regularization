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
parser.add_argument('--decoder',
                    help='Reconstruction loss',
                    dest='decoder', action='store_true')
parser.add_argument('--mask',
                    help='Mask representation',
                    dest='mask', action='store_true')
parser.add_argument('--conv',
                    help='Use convs for reconstruction',
                    dest='conv', action='store_true')
parser.add_argument('--regularizer',
                    help='Regularizer to use',
                    dest='regularizer', type=str)
parser.add_argument('--regularizer-weight',
                    help='Weight corresponding to regularizer',
                    dest='regularizer_weight', type=float)
parser.add_argument('--epochs',
                    help='Training epochs',
                    dest='epochs', type=int, required=True)
parser.add_argument('--lr',
                    help='Learning rate',
                    dest='lr', type=float, default=1e-4)
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


def main(options):
    start = time.time()

    assert options.model in MODELS

    logging.info('Creating data generators.')
    data_gen = DATA_GEN[options.data]
    train_gen = data_gen('train', batch_size=options.batch_size, decoder=options.decoder, shuffle=True)
    val_gen = data_gen('val', batch_size=options.batch_size, decoder=options.decoder)
    test_gen = data_gen('test', batch_size=1, decoder=options.decoder)
    if options.decoder:
        pred_gen = data_gen('pred', batch_size=1, decoder=options.decoder, include_label=False)

    logging.info('Creating model.')
    m = MODELS[options.model](options.name,
                              CLASSES[options.data],
                              IMAGE_SHAPE[options.data],
                              options.lr,
                              decoder=options.decoder,
                              mask=options.mask,
                              conv=options.conv,
                              regularizer=options.regularizer,
                              regularizer_weight=options.regularizer_weight,
                              save_freq=options.save_freq,
                              tensorboard=options.tensorboard,
                              filename=options.model_file)

    logging.info('Training model.')
    m.train(train_gen, val_gen, options.epochs)

    # logging.info('Testing model.')
    # metrics = m.test(test_gen)
    # logging.info(metrics)

    if options.decoder:
        logging.info('Making predictions.')
        os.makedirs(f'data/{options.name}/', exist_ok=True)
        preds = m.predict(pred_gen)[1]
        for i in range(preds.shape[0]):
            util.save_img(pred_gen[i][0][0], f'data/{options.name}/{str(i).zfill(4)}_true.png')
            util.save_img(preds[i], f'data/{options.name}/{str(i).zfill(4)}.png')

    end = time.time()
    logging.info(f'total time: {datetime.timedelta(seconds=(end - start))}')


if __name__ == '__main__':
    main(options)

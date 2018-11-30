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
parser.add_argument('--routings',
                    help='Routing iterations',
                    dest='routings', type=int, default=3)
parser.add_argument('--model-file',
                    help='Pretrained model file',
                    dest='model_file', type=str)

options = parser.parse_args()

import os
import data
import models
import numpy as np
import util


MODELS = {
    'conv': models.ConvNet,
    'ae': models.Autoencoder,
    'caps': models.CapsNet,
}

DATA_GEN = {
    'cubes': data.CubeGenerator,
    'mnist': data.MNISTGenerator,
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
    'caps': {
        'cubes': 'pose',
        'mnist': 'digit',
    },
}

CLASSES = {
    'cubes': 3,
    'mnist': 10,
}

IMAGE_SHAPE = {
    'cubes': (32, 32, 3),
    'mnist': (32, 32, 1),
}

LOSS = {
    'cubes': 'mse',
    'mnist': 'categorical_crossentropy',
}


def get_gen_args(data, split):
    args = {}
    if data == 'cubes':
        args['image_size'] = 32
        args['n'] = { 'train': 1000, 'val': 100, 'pred': 10, 'test': 10 }[split]
    elif data == 'mnist':
        partition = split if split != 'pred' else 'test'
        args['partition'] = partition
    else:
        raise ValueError(f'{data} is not a valid dataset.')
    return args


def main(options):
    assert options.model in MODELS

    logging.info('Creating data generators.')
    data_gen = DATA_GEN[options.data]
    label_type = LABEL[options.model][options.data]
    train_gen = data_gen(batch_size=options.batch_size, label_type=label_type, **get_gen_args(options.data, 'train'))
    val_gen = data_gen(batch_size=options.batch_size, label_type=label_type, **get_gen_args(options.data, 'val'))
    pred_gen = data_gen(batch_size=1, shuffle=False, **get_gen_args(options.data, 'pred'))
    test_gen = data_gen(batch_size=1, label_type=label_type, shuffle=False, **get_gen_args(options.data, 'test'))

    logging.info('Creating model.')
    m = MODELS[options.model](options.name,
                              CLASSES[options.data],
                              IMAGE_SHAPE[options.data],
                              LOSS[options.data],
                              options.tensorboard,
                              routings=options.routings,
                              filename=options.model_file)

    logging.info('Training model.')
    m.train(train_gen, val_gen, options.epochs)

    logging.info('Making predictions.')
    preds = m.predict(pred_gen)
    os.makedirs(f'data/{options.name}/', exist_ok=True)
    for i in range(preds.shape[0]):
        util.save_img(pred_gen[i][0], f'data/{options.name}/{str(i).zfill(4)}_true.png')
        if options.model == 'ae':
            util.save_img(preds[i], f'data/{options.name}/{str(i).zfill(4)}.png')
        elif options.data == 'cubes':
            util.save_img(util.draw_cube(util.rotation_matrix(*preds[i])), f'data/{options.name}/{str(i).zfill(4)}.png')
        elif options.data == 'mnist':
            preds = np.argmax(preds, axis=0)
            logging.info('Predictions: ', preds)

    logging.info('Testing model.')
    metrics = m.test(test_gen)
    logging.info(metrics)


if __name__ == '__main__':
    main(options)

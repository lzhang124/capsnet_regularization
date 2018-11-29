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
parser.add_argument('--epochs',
                    help='Training epochs',
                    dest='epochs', type=int, required=True)
parser.add_argument('--name',
                    help='Name of model',
                    dest='name', type=str, required=True)
parser.add_argument('--tensorboard',
                    help='Enable tensorboard',
                    dest='tensorboard', action='store_true')
options = parser.parse_args()

## START capsnet
if __name__ == "__main__":
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # define model
    model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                  routings=args.routings)
    model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        manipulate_latent(manipulate_model, (x_test, y_test), args)
        test(model=eval_model, data=(x_test, y_test), args=args)
## END CapsNet


import os
import cubes
import mnist
import models
import util


MODELS = {
    'conv': models.ConvNet,
    'ae': models.Autoencoder,
    'caps': models.CapsNet,
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
    'caps': {
        'cubes': 'pose',
        'mnist': 'digit',
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
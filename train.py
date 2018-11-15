import cubes
import models


def main():
    train_gen = cubes.CubeGenerator(100, label_type='pose')
    val_gen = cubes.CubeGenerator(10, label_type='pose')
    test_gen = cubes.CubeGenerator(10, label_type='pose')

    m = models.ConvNet('base_conv_net_cubes')
    m.compile()
    m.train(train_gen, val_gen, 10)


if __name__ == '__main__':
    main()
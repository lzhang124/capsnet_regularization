for model in "$@"
do
    if [ $model = "caps" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_mnist -o caps_mnist.out -e caps_mnist.err python train.py --model caps --data mnist --epochs 15 --batch-size 100 --name caps_mnist_15 --save-freq 1 --tensorboard &
    elif [ $model = "l1" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_mnist_l1 -o caps_mnist_l1.out -e caps_mnist_l1.err python train.py --model caps --data mnist --epochs 15 --batch-size 100 --regularizers l1 --regularizer-weights 0.01 --name caps_mnist_15_l1 --save-freq 1 --tensorboard &
    elif [ $model = "l2" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_mnist_l2 -o caps_mnist_l2.out -e caps_mnist_l2.err python train.py --model caps --data mnist --epochs 15 --batch-size 100 --regularizers l2 --regularizer-weights 0.01 --name caps_mnist_15_l2 --save-freq 1 --tensorboard &
    elif [ $model = "l21" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_mnist_l21 -o caps_mnist_l21.out -e caps_mnist_l21.err python train.py --model caps --data mnist --epochs 15 --batch-size 100 --regularizers l21 --regularizer-weights 0.01 --name caps_mnist_15_l21 --save-freq 1 --tensorboard &
    elif [ $model = "op" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_mnist_op -o caps_mnist_op.out -e caps_mnist_op.err python train.py --model caps --data mnist --epochs 15 --batch-size 100 --regularizers operator_norm --regularizer-weights 0.01 --name caps_mnist_15_op --save-freq 1 --tensorboard &
    elif [ $model = "conv" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J convcaps_mnist -o convcaps_mnist.out -e convcaps_mnist.err python train.py --model convcaps --data mnist --epochs 15 --batch-size 20 --name convcaps_mnist_15 --save-freq 1 --tensorboard &
    elif [ $model = "full" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J fullcaps_mnist -o fullcaps_mnist.out -e fullcaps_mnist.err python train.py --model fullcaps --data mnist --epochs 15 --batch-size 20 --name fullcaps_mnist_15 --save-freq 1 --tensorboard &
    else
        echo "invalid model type"
    fi
done



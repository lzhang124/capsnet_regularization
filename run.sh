if [ $1 = "rm" ]
then
    shift
    rm -rf logs/*
    rm -rf models/*
    rm -rf data/*
    sleep 1
fi

for model in "$@"
do
    if [ $model = "caps" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps -o caps.out -e caps.err python train.py --model caps --data cifar --epochs 30 --batch-size 100 --name caps_30 --save-freq 1 --tensorboard &
    elif [ $model = "l1" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_l1_0.01 -o caps_l1_0.01.out -e caps_l1_0.01.err python train.py --model caps --data cifar --epochs 30 --batch-size 100 --regularizer l1 --regularizer-weight 0.01 --name caps_30_l1_0.01 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_l1_0.001 -o caps_l1_0.001.out -e caps_l1_0.001.err python train.py --model caps --data cifar --epochs 30 --batch-size 100 --regularizer l1 --regularizer-weight 0.001 --name caps_30_l1_0.001 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_l1_0.0001 -o caps_l1_0.0001.out -e caps_l1_0.0001.err python train.py --model caps --data cifar --epochs 30 --batch-size 100 --regularizer l1 --regularizer-weight 0.0001 --name caps_30_l1_0.0001 --save-freq 1 --tensorboard &

    elif [ $model = "l2" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_l2_0.01 -o caps_l2_0.01.out -e caps_l2_0.01.err python train.py --model caps --data cifar --epochs 30 --batch-size 100 --regularizer l2 --regularizer-weight 0.01 --name caps_30_l2_0.01 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_l2_0.001 -o caps_l2_0.001.out -e caps_l2_0.001.err python train.py --model caps --data cifar --epochs 30 --batch-size 100 --regularizer l2 --regularizer-weight 0.001 --name caps_30_l2_0.001 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_l2_0.0001 -o caps_l2_0.0001.out -e caps_l2_0.0001.err python train.py --model caps --data cifar --epochs 30 --batch-size 100 --regularizer l2 --regularizer-weight 0.0001 --name caps_30_l2_0.0001 --save-freq 1 --tensorboard &

    elif [ $model = "l21" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_l21_0.01 -o caps_l21_0.01.out -e caps_l21_0.01.err python train.py --model caps --data cifar --epochs 30 --batch-size 100 --regularizer l21 --regularizer-weight 0.01 --name caps_30_l21_0.01 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_l21_0.001 -o caps_l21_0.001.out -e caps_l21_0.001.err python train.py --model caps --data cifar --epochs 30 --batch-size 100 --regularizer l21 --regularizer-weight 0.001 --name caps_30_l21_0.001 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_l21_0.0001 -o caps_l21_0.0001.out -e caps_l21_0.0001.err python train.py --model caps --data cifar --epochs 30 --batch-size 100 --regularizer l21 --regularizer-weight 0.0001 --name caps_30_l21_0.0001 --save-freq 1 --tensorboard &

    elif [ $model = "op" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_op_0.01 -o caps_op_0.01.out -e caps_op_0.01.err python train.py --model caps --data cifar --epochs 30 --batch-size 100 --regularizer operator_norm --regularizer-weight 0.01 --name caps_30_op_0.01 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_op_0.001 -o caps_op_0.001.out -e caps_op_0.001.err python train.py --model caps --data cifar --epochs 30 --batch-size 100 --regularizer operator_norm --regularizer-weight 0.001 --name caps_30_op_0.001 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_op_0.0001 -o caps_op_0.0001.out -e caps_op_0.0001.err python train.py --model caps --data cifar --epochs 30 --batch-size 100 --regularizer operator_norm --regularizer-weight 0.0001 --name caps_30_op_0.0001 --save-freq 1 --tensorboard &
    elif [ $model = "conv" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J convcaps -o convcaps.out -e convcaps.err python train.py --model convcaps --data cifar --epochs 5 --batch-size 100 --name convcaps_5 --save-freq 1 --tensorboard &
    elif [ $model = "full" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J fullcaps -o fullcaps.out -e fullcaps.err python train.py --model fullcaps --data cifar --epochs 5 --batch-size 100 --name fullcaps_5 --save-freq 1 --tensorboard &
    elif [ $model = "tensorboard" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 tensorboard --logdir=logs/ --port=6120
    else
        echo "invalid model type"
    fi
done



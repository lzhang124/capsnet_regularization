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
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J convcaps_1 -o convcaps_1.out -e convcaps_1.err python train.py --model convcaps --data cifar --epochs 30 --lr 1e-1 --batch-size 100 --name convcaps_30_1 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J convcaps_2 -o convcaps_2.out -e convcaps_2.err python train.py --model convcaps --data cifar --epochs 30 --lr 1e-2 --batch-size 100 --name convcaps_30_2 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J convcaps_3 -o convcaps_3.out -e convcaps_3.err python train.py --model convcaps --data cifar --epochs 30 --lr 1e-3 --batch-size 100 --name convcaps_30_3 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J convcaps_4 -o convcaps_4.out -e convcaps_4.err python train.py --model convcaps --data cifar --epochs 30 --lr 1e-4 --batch-size 100 --name convcaps_30_4 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J convcaps_5 -o convcaps_5.out -e convcaps_5.err python train.py --model convcaps --data cifar --epochs 30 --lr 1e-5 --batch-size 100 --name convcaps_30_5 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J convcaps_6 -o convcaps_6.out -e convcaps_6.err python train.py --model convcaps --data cifar --epochs 30 --lr 1e-6 --batch-size 100 --name convcaps_30_6 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J convcaps_7 -o convcaps_7.out -e convcaps_7.err python train.py --model convcaps --data cifar --epochs 30 --lr 1e-7 --batch-size 100 --name convcaps_30_7 --save-freq 1 --tensorboard &
    elif [ $model = "full" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J fullcaps -o fullcaps.out -e fullcaps.err python train.py --model fullcaps --data cifar --epochs 30 --batch-size 100 --name fullcaps_30 --save-freq 1 --tensorboard &
    else
        echo "invalid model type"
    fi
done



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
    if [ $model = "conv" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J conv -o conv.out -e conv.err python train.py --model conv --data cifar --epochs 30 --batch-size 100 --name conv_30 --save-freq 1 --tensorboard &
    elif [ $model = "caps" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps -o caps.out -e caps.err python train.py --model caps --data cifar --epochs 30 --batch-size 100 --name caps_30 --save-freq 1 --tensorboard &
    elif [ $model = "caps_r" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_r -o caps_r.out -e caps_r.err python train.py --model caps --data cifar --epochs 50 --batch-size 100 --name caps_r_50 --decoder --save-freq 1 --tensorboard &
    elif [ $model = "caps_rm" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_rm -o caps_rm.out -e caps_rm.err python train.py --model caps --data cifar --epochs 50 --batch-size 100 --name caps_rm_50 --decoder --mask --save-freq 1 --tensorboard &
    elif [ $model = "caps_rc" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_rc -o caps_rc.out -e caps_rc.err python train.py --model caps --data cifar --epochs 50 --batch-size 100 --name caps_rc_50 --decoder --conv --save-freq 1 --tensorboard &
    elif [ $model = "caps_rmc" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_rmc -o caps_rmc.out -e caps_rmc.err python train.py --model caps --data cifar --epochs 50 --batch-size 100 --name caps_rmc_50 --decoder --mask --conv --save-freq 1 --tensorboard &
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
    elif [ $model = "convcaps" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J convcaps -o convcaps.out -e convcaps.err python train.py --model convcaps --data cifar --epochs 30 --batch-size 100 --name convcaps_30 --save-freq 1 --tensorboard &
    elif [ $model = "fullcaps" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J fullcaps -o fullcaps.out -e fullcaps.err python train.py --model fullcaps --data cifar --epochs 30 --batch-size 100 --name fullcaps_30 --save-freq 1 --tensorboard &
    else
        echo "invalid model type"
    fi
done



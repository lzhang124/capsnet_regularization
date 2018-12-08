if [ $1 = "rm" ]
then
    shift
    rm -rf logs/*
    rm -rf models/*
    rm -rf data/*
    sleep 1
fi

data=$1
shift

for model in "$@"
do
    if [ $model = "caps" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps -o caps.out -e caps.err python train.py --model caps --data ${data} --epochs 30 --batch-size 100 --name caps_30 --save-freq 1 --tensorboard &
    elif [ $model = "l1" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_l1_5 -o caps_l1_5.out -e caps_l1_5.err python train.py --model caps --data ${data} --epochs 30 --batch-size 100 --regularizer l1 --regularizer-weight 5 --name caps_30_l1_5 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_l1_1 -o caps_l1_1.out -e caps_l1_1.err python train.py --model caps --data ${data} --epochs 30 --batch-size 100 --regularizer l1 --regularizer-weight 1 --name caps_30_l1_1 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_l1_10 -o caps_l1_10.out -e caps_l1_10.err python train.py --model caps --data ${data} --epochs 30 --batch-size 100 --regularizer l1 --regularizer-weight 10 --name caps_30_l1_10 --save-freq 1 --tensorboard &

    elif [ $model = "l2" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_l2_5 -o caps_l2_5.out -e caps_l2_5.err python train.py --model caps --data ${data} --epochs 30 --batch-size 100 --regularizer l2 --regularizer-weight 5 --name caps_30_l2_5 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_l2_1 -o caps_l2_1.out -e caps_l2_1.err python train.py --model caps --data ${data} --epochs 30 --batch-size 100 --regularizer l2 --regularizer-weight 0.000001 --name caps_30_l2_1 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_l2_10 -o caps_l2_10.out -e caps_l2_10.err python train.py --model caps --data ${data} --epochs 30 --batch-size 100 --regularizer l2 --regularizer-weight 10 --name caps_30_l2_10 --save-freq 1 --tensorboard &

    elif [ $model = "l21" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_l21_5 -o caps_l21_5.out -e caps_l21_5.err python train.py --model caps --data ${data} --epochs 30 --batch-size 100 --regularizer l21 --regularizer-weight 5 --name caps_30_l21_5 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_l21_1 -o caps_l21_1.out -e caps_l21_1.err python train.py --model caps --data ${data} --epochs 30 --batch-size 100 --regularizer l21 --regularizer-weight 1 --name caps_30_l21_1 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_l21_10 -o caps_l21_10.out -e caps_l21_10.err python train.py --model caps --data ${data} --epochs 30 --batch-size 100 --regularizer l21 --regularizer-weight 10 --name caps_30_l21_10 --save-freq 1 --tensorboard &

    elif [ $model = "op" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_op_5 -o caps_op_5.out -e caps_op_5.err python train.py --model caps --data ${data} --epochs 30 --batch-size 100 --regularizer operator_norm --regularizer-weight 5 --name caps_30_op_5 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_op_1 -o caps_op_1.out -e caps_op_1.err python train.py --model caps --data ${data} --epochs 30 --batch-size 100 --regularizer operator_norm --regularizer-weight 1 --name caps_30_op_1 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_op_10 -o caps_op_10.out -e caps_op_10.err python train.py --model caps --data ${data} --epochs 30 --batch-size 100 --regularizer operator_norm --regularizer-weight 10 --name caps_30_op_10 --save-freq 1 --tensorboard &
    elif [ $model = "conv" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J convcaps -o convcaps.out -e convcaps.err python train.py --model convcaps --data ${data} --epochs 5 --lr 1e-7 --batch-size 100 --name convcaps_5 --save-freq 1 --tensorboard &
    elif [ $model = "full" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J fullcaps -o fullcaps.out -e fullcaps.err python train.py --model fullcaps --data ${data} --epochs 5 --lr 1e-7 --batch-size 100 --name fullcaps_5 --save-freq 1 --tensorboard &
    elif [ $model = "tensorboard" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 tensorboard --logdir=logs/ --port=6120
    else
        echo "invalid model type"
    fi
done



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
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_${data} -o caps_${data}.out -e caps_${data}.err python train.py --model caps --data ${data} --epochs 30 --batch-size 100 --name caps_${data}_30 --save-freq 1 --tensorboard &
    elif [ $model = "l1" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_${data}_l1_1 -o caps_${data}_l1_1.out -e caps_${data}_l1_1.err python train.py --model caps --data ${data} --epochs 30 --batch-size 100 --regularizer l1 --regularizer-weight 0.1 --name caps_${data}_30_l1_1 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_${data}_l1_01 -o caps_${data}_l1_01.out -e caps_${data}_l1_01.err python train.py --model caps --data ${data} --epochs 30 --batch-size 100 --regularizer l1 --regularizer-weight 0.01 --name caps_${data}_30_l1_01 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_${data}_l1_001 -o caps_${data}_l1_001.out -e caps_${data}_l1_001.err python train.py --model caps --data ${data} --epochs 30 --batch-size 100 --regularizer l1 --regularizer-weight 0.001 --name caps_${data}_30_l1_001 --save-freq 1 --tensorboard &

    elif [ $model = "l2" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_${data}_l2_1 -o caps_${data}_l2_1.out -e caps_${data}_l2_1.err python train.py --model caps --data ${data} --epochs 30 --batch-size 100 --regularizer l2 --regularizer-weight 0.000001 --name caps_${data}_30_l2_1 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_${data}_l2_01 -o caps_${data}_l2_01.out -e caps_${data}_l2_01.err python train.py --model caps --data ${data} --epochs 30 --batch-size 100 --regularizer l2 --regularizer-weight 0.01 --name caps_${data}_30_l2_01 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_${data}_l2_001 -o caps_${data}_l2_001.out -e caps_${data}_l2_001.err python train.py --model caps --data ${data} --epochs 30 --batch-size 100 --regularizer l2 --regularizer-weight 0.001 --name caps_${data}_30_l2_001 --save-freq 1 --tensorboard &

    elif [ $model = "l21" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_${data}_l21_1 -o caps_${data}_l21_1.out -e caps_${data}_l21_1.err python train.py --model caps --data ${data} --epochs 30 --batch-size 100 --regularizer l21 --regularizer-weight 0.1 --name caps_${data}_30_l21_1 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_${data}_l21_01 -o caps_${data}_l21_01.out -e caps_${data}_l21_01.err python train.py --model caps --data ${data} --epochs 30 --batch-size 100 --regularizer l21 --regularizer-weight 0.01 --name caps_${data}_30_l21_01 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_${data}_l21_001 -o caps_${data}_l21_001.out -e caps_${data}_l21_001.err python train.py --model caps --data ${data} --epochs 30 --batch-size 100 --regularizer l21 --regularizer-weight 0.001 --name caps_${data}_30_l21_001 --save-freq 1 --tensorboard &

    elif [ $model = "op" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_${data}_op_1 -o caps_${data}_op_1.out -e caps_${data}_op_1.err python train.py --model caps --data ${data} --epochs 30 --batch-size 100 --regularizer operator_norm --regularizer-weight 0.1 --name caps_${data}_30_op_1 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_${data}_op_01 -o caps_${data}_op_01.out -e caps_${data}_op_01.err python train.py --model caps --data ${data} --epochs 30 --batch-size 100 --regularizer operator_norm --regularizer-weight 0.01 --name caps_${data}_30_op_01 --save-freq 1 --tensorboard &
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_${data}_op_001 -o caps_${data}_op_001.out -e caps_${data}_op_001.err python train.py --model caps --data ${data} --epochs 30 --batch-size 100 --regularizer operator_norm --regularizer-weight 0.001 --name caps_${data}_30_op_001 --save-freq 1 --tensorboard &
    elif [ $model = "conv" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J convcaps_${data} -o convcaps_${data}.out -e convcaps_${data}.err python train.py --model convcaps --data ${data} --epochs 5 --batch-size 100 --name convcaps_${data}_5 --save-freq 1 --tensorboard &
    elif [ $model = "full" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J fullcaps_${data} -o fullcaps_${data}.out -e fullcaps_${data}.err python train.py --model fullcaps --data ${data} --epochs 5 --batch-size 100 --name fullcaps_${data}_5 --save-freq 1 --tensorboard &
    elif [ $model = "tensorboard" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 tensorboard --logdir=logs/ --port=6120
    else
        echo "invalid model type"
    fi
done



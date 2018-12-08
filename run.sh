if [ $1 = "rm" ]
then
    shift
    rm -rf logs/*
    rm -rf models/*
    rm -rf data/*
fi

data=$1
shift

for model in "$@"
do
    if [ $model = "caps" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_${data} -o caps_${data}.out -e caps_${data}.err python train.py --model caps --data ${data} --epochs 15 --batch-size 100 --name caps_${data}_15 --save-freq 1 --tensorboard &
    elif [ $model = "l1" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_${data}_l1 -o caps_${data}_l1.out -e caps_${data}_l1.err python train.py --model caps --data ${data} --epochs 15 --batch-size 100 --regularizer l1 --regularizer-weight 0 --name caps_${data}_15_l1 --save-freq 1 --tensorboard &
    elif [ $model = "l2" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_${data}_l2 -o caps_${data}_l2.out -e caps_${data}_l2.err python train.py --model caps --data ${data} --epochs 15 --batch-size 100 --regularizer l2 --regularizer-weight 0 --name caps_${data}_15_l2 --save-freq 1 --tensorboard &
    elif [ $model = "l21" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_${data}_l21 -o caps_${data}_l21.out -e caps_${data}_l21.err python train.py --model caps --data ${data} --epochs 15 --batch-size 100 --regularizer l21 --regularizer-weight 0 --name caps_${data}_15_l21 --save-freq 1 --tensorboard &
    elif [ $model = "op" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_${data}_op -o caps_${data}_op.out -e caps_${data}_op.err python train.py --model caps --data ${data} --epochs 15 --batch-size 100 --regularizer operator_norm --regularizer-weight 0 --name caps_${data}_15_op --save-freq 1 --tensorboard &
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



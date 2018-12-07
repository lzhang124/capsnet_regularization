data=$1
shift

for model in "$@"
do
    if [ $model = "caps" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_${data} -o caps_${data}.out -e caps_${data}.err python train.py --model caps --data ${data} --epochs 50 --batch-size 8 --name caps_${data}_50 --save-freq 1 --tensorboard &
    elif [ $model = "l1" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_${data}_l1 -o caps_${data}_l1.out -e caps_${data}_l1.err python train.py --model caps --data ${data} --epochs 50 --batch-size 8 --regularizers l1 --regularizer-weights 0.01 --name caps_${data}_50_l1 --save-freq 1 --tensorboard &
    elif [ $model = "l2" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_${data}_l2 -o caps_${data}_l2.out -e caps_${data}_l2.err python train.py --model caps --data ${data} --epochs 50 --batch-size 8 --regularizers l2 --regularizer-weights 0.01 --name caps_${data}_50_l2 --save-freq 1 --tensorboard &
    elif [ $model = "l21" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_${data}_l21 -o caps_${data}_l21.out -e caps_${data}_l21.err python train.py --model caps --data ${data} --epochs 50 --batch-size 8 --regularizers l21 --regularizer-weights 0.01 --name caps_${data}_50_l21 --save-freq 1 --tensorboard &
    elif [ $model = "op" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J caps_${data}_op -o caps_${data}_op.out -e caps_${data}_op.err python train.py --model caps --data ${data} --epochs 50 --batch-size 8 --regularizers operator_norm --regularizer-weights 0.01 --name caps_${data}_50_op --save-freq 1 --tensorboard &
    elif [ $model = "conv" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J convcaps_${data} -o convcaps_${data}.out -e convcaps_${data}.err python train.py --model convcaps --data ${data} --epochs 50 --batch-size 8 --name convcaps_${data}_50 --save-freq 1 --tensorboard &
    elif [ $model = "full" ]
    then
        srun -p gpu -t 10:00:00 --mem-per-cpu 1 --gres=gpu:1 -J fullcaps_${data} -o fullcaps_${data}.out -e fullcaps_${data}.err python train.py --model fullcaps --data ${data} --epochs 50 --batch-size 8 --name fullcaps_${data}_50 --save-freq 1 --tensorboard &
    else
        echo "invalid model type"
    fi
done



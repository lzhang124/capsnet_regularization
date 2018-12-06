slurm -n caps_mnist python train.py --model caps --data mnist --epochs 15 --batch-size 100 --name caps_mnist_15 --save-freq 1 --tensorboard &
slurm -n caps_mnist_l1 python train.py --model caps --data mnist --epochs 15 --batch-size 100 --regularizers l1 --regularizer-weights 0.01 --name caps_mnist_15_l1 --save-freq 1 --tensorboard &
slurm -n caps_mnist_l2 python train.py --model caps --data mnist --epochs 15 --batch-size 100 --regularizers l2 --regularizer-weights 0.01 --name caps_mnist_15_l2 --save-freq 1 --tensorboard &
slurm -n caps_mnist_l21 python train.py --model caps --data mnist --epochs 15 --batch-size 100 --regularizers l21 --regularizer-weights 0.01 --name caps_mnist_15_l21 --save-freq 1 --tensorboard &
slurm -n caps_mnist_op python train.py --model caps --data mnist --epochs 15 --batch-size 100 --regularizers operator_norm --regularizer-weights 0.01 --name caps_mnist_15_op --save-freq 1 --tensorboard &
slurm -n convcaps_mnist python train.py --model convcaps --data mnist --epochs 15 --batch-size 100 --name convcaps_mnist_15 --save-freq 1 --tensorboard &
slurm -n fullcaps_mnist python train.py --model fullcaps --data mnist --epochs 15 --batch-size 100 --name fullcaps_mnist_15 --save-freq 1 --tensorboard &

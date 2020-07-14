# DO NOT CHANGE THIS
DIR="/home/jz288/cat"
cd $DIR


#python train_fast.py --gpu 1,2,3,0 --epochs 20 --lr-sch cyclic --seed 0 --alpha 10
python train_fast.py --gpu 1,2,3,0 --epochs 20 --lr-sch cyclic --seed 1 --alpha 10
python train_fast.py --gpu 1,2,3,0 --epochs 20 --lr-sch cyclic --seed 2 --alpha 10

python train_fast.py --gpu 1,2,3,0 --epochs 20 --lr-sch cyclic --seed 0 --alpha 8
python train_fast.py --gpu 1,2,3,0 --epochs 20 --lr-sch cyclic --seed 1 --alpha 8
python train_fast.py --gpu 1,2,3,0 --epochs 20 --lr-sch cyclic --seed 2 --alpha 8

python train_fast.py --gpu 1,2,3,0 --epochs 20 --lr-sch cyclic --seed 0 --alpha 4
python train_fast.py --gpu 1,2,3,0 --epochs 20 --lr-sch cyclic --seed 1 --alpha 4
python train_fast.py --gpu 1,2,3,0 --epochs 20 --lr-sch cyclic --seed 2 --alpha 4

python train_fast.py --gpu 1,2,3,0 --epochs 20 --lr-sch cyclic --seed 0 --alpha 12
python train_fast.py --gpu 1,2,3,0 --epochs 20 --lr-sch cyclic --seed 1 --alpha 12
python train_fast.py --gpu 1,2,3,0 --epochs 20 --lr-sch cyclic --seed 2 --alpha 12

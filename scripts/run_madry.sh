# DO NOT CHANGE THIS
DIR="/home/jz288/cat"
cd $DIR

GPU="4,5,6,7"

#python train_madry.py --gpu $GPU --epochs 20 --lr-sch multistep --steps 10 --seed 0 --batch-size 256
#python train_madry.py --gpu 4,5,6,7 --epochs 100 --lr-sch multistep --steps 10 --seed 0 --batch-size 256 --amp
#python train_madry.py --gpu 4,5,6,7 --epochs 100 --lr-sch multistep --steps 5 --seed 0 --batch-size 256 --amp

#python train_madry.py --gpu 4,5,6,7 --epochs 100 --lr-sch multistep --steps 2 --seed 0 --batch-size 256 --amp
python train_madry.py --gpu 4,5,6,7 --epochs 200 --lr-sch multistep --seed 1 --batch-size 256 --amp

# DO NOT CHANGE THIS
DIR=".."
cd $DIR

GPU="3"

python train_natural.py --gpu $GPU --epochs 200 --lr-sch multistep --seed 0 --batch-size 256 --amp

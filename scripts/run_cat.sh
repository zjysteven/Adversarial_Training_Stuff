# DO NOT CHANGE THIS
DIR=".."
cd $DIR



GPU="4,5,6,7"

python train_cat.py --gpu 4,5,6,7 --epochs 200 --lr-sch multistep --sch-intervals 80 140 180 \
    --seed 233 --batch-size 256 --amp

python train_cat.py --gpu 4,5,6,7 --epochs 200 --lr-sch multistep --sch-intervals 80 140 180 \
    --seed 233 --batch-size 256 --amp --depth 34
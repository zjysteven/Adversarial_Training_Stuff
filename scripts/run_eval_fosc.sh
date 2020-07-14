# DO NOT CHANGE THIS
DIR="/home/jz288/cat"
cd $DIR


GPU="4,5,6,7"
MODELPATH="checkpoints/wrn16/madry/seed_1/\
epochs_100_batch_256_lr_multistep_alpha_2_steps_10_O1/state_dicts"
#MODELPATH="checkpoints/wrn16/natural/seed_0/\
#epochs_200_batch_256_lr_multistep_O1/state_dicts"

#declare -a arr=(50 51 60 65 70)

for i in $(seq 3 2 99); do
#for i in "${arr[@]}"; do
    python evaluate_fosc.py --gpu $GPU --batch-size 4000 \
        --model-file ${MODELPATH}/epoch_${i}.pth \
        --alpha 2 --steps 10 --save
    python evaluate_fosc.py --gpu $GPU --batch-size 4000 \
        --model-file ${MODELPATH}/epoch_${i}.pth \
        --alpha 2 --steps 10 --save --trainset
done

#python evaluate_fosc.py --gpu $GPU --batch-size 1000 \
#    --model-file ${MODELPATH}/epoch_51.pth \
#    --alpha 2 --steps 10 --trainset

#python evaluate_fosc.py --gpu $GPU --batch-size 1000 \
#    --model-file ${MODELPATH}/epoch_60.pth \
#    --alpha 2 --steps 10 --trainset

#python evaluate_fosc.py --gpu $GPU --batch-size 1000 \
#    --model-file ${MODELPATH}/epoch_50.pth \
#    --alpha 2 --steps 10 --save


#for i in $(seq 10 10 100); do
#    python evaluate_wbox.py --gpu $GPU --benchmark --save-adv --save-to-csv --random-start 1 \
#        --model-file ${FILEPATH}/epoch_${i}.pth --steps 100
#    python evaluate_wbox.py --gpu $GPU --benchmark --save-adv --save-to-csv --random-start 1 \
#        --model-file ${FILEPATH}/epoch_${i}.pth --steps 100 --trainset
#done


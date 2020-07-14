# DO NOT CHANGE THIS
DIR=".."
cd $DIR


FILEPATH="checkpoints/wrn16/madry/seed_1/epochs_100_batch_256_lr_multistep_alpha_2_steps_10_eval_O1/state_dicts"
GPU="4,5,6,7"


for i in $(seq 10 10 100); do
    python evaluate_wbox.py --gpu $GPU --benchmark --save-adv --save-to-csv --random-start 1 \
        --model-file ${FILEPATH}/epoch_${i}.pth --steps 100
    python evaluate_wbox.py --gpu $GPU --benchmark --save-adv --save-to-csv --random-start 1 \
        --model-file ${FILEPATH}/epoch_${i}.pth --steps 100 --trainset
done


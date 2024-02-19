for BASE_MODEL in google/flan-t5-base google/flan-t5-large
do
    for TASK_NAME in isear rt-polarity
    do
        for SEED in 0 1 2
        do
            export SAVE_CHECKPOINT=yes
            export TASK_NAME
            export SEED
            export MODEL
            export TRAIN_SAMPLES=1200
            export RETRAIN_FREQ=100
            export BUDGET=1000
            PART=csd3
            sbatch --export=ALL scripts/sub_$PART.sh
        done
    done
done
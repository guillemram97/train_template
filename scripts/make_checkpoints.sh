for TASK_NAME in isear rt-polarity openbook fever
do
    for SEED in 0
    do
        export TASK_NAME
        export SEED
        PART=csd3
        export PART
        sbatch --export=ALL scripts/sub_$PART.sh
    done
done
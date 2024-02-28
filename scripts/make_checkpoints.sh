for TASK in fever openbook isear rt-polarity cr sst2
do
    for MODEL in Mistral-Instruct-4b #Llama-7b-4b Llama-13b-4b llama-1-30b-4b falcon-40b-4b #llama-1-65b-4b Llama-70b-4b
    do
        for SEED in 0
        do
            export TASK_NAME=${TASK}_prob_${MODEL}
            export SEED
            PART=csd3
            export PART
            sbatch --export=ALL scripts/sub_$PART.sh
        done
    done
done
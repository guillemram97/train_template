for TASK in babi_qa natural_qa wikifact
do
    #for MODEL in Mixtral-Instruct-4b Mistral-Instruct-4b #Llama-7b-4b Llama-13b-4b llama-1-30b-4b falcon-40b-4b #llama-1-65b-4b Llama-70b-4b
    #do
     #   for SEED in 0
     #   do
	export TASK_NAME=${TASK}
	export SEED=0
	PART=csd3
	export PART
	sbatch --export=ALL scripts/sub_$PART.sh
        #done
   # done
    #export TASK_NAME=${TASK}_prob_${MODEL}
    #sbatch --export=ALL scripts/sub_$PART.sh
done

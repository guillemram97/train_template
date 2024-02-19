#source scripts/cluster.sh
export TRAIN_SAMPLES=10000
export TARGET=llm
export DATA_PATH=/work/dc007/dc007/cs-rami1/data
export PART=csd3
export BASE_MODEL=t5-base

# HE ENVIAT MASSES JOBS, AIXI HO HE DEIXAT!
for RETRAIN_FREQ in 1000
do
    for SEED in 0 1 2
    do
        for BUDGET in 1000 1500 2000 2500 3000 3500
        do  # cr ag_news isear_llama rt-polarity_llama isear_mistral rt-polarity_mistral
            for TASK_NAME in fever_mistral openbook_mistral #rt-polarity_mistral isear_mistral rt-polarity_llama #fever_llama openbook_llama fever_mistral openbook_mistral
            do 
                for STRATEGY in b2 #els deixo pel feturo!! 108 jobs funciona be
                do
                    export N_INIT=1000
                    export TASK_NAME
                    export STRATEGY
                    export BUDGET
                    export RETRAIN_FREQ
                    export TAGS=CIRRUS_ARR_CORRECT_BATCH
                    export CHECKPOINT=${SEED}_${N_INIT}
                    export SEED

                    if [ $STRATEGY == "b2" ]
                    then
                        export P_STRAT=0
                        sbatch --export=ALL scripts/sub_$PART.sh
                    fi
                    if [ $STRATEGY == "BT" ]
                    then 
                        for P_STRAT in 5
                        do
                            export P_STRAT
                            sbatch --export=ALL scripts/sub_$PART.sh
                        done
                    fi
                    if [ $STRATEGY == "MV" ]
                    then
                        export P_STRAT=3
                        sbatch --export=ALL scripts/sub_$PART.sh
                    fi
                    if [ $STRATEGY == "EN" ]
                    then
                        for P_STRAT in 0.5
                        do
                            export P_STRAT
                            sbatch --export=ALL scripts/sub_$PART.sh
                        done
                    fi 
                    if [ $STRATEGY == "CS" ]
                    then
                        for P_STRAT in 0.9
                        do
                            export P_STRAT
                            export EMBED=t5
                            sbatch --export=ALL scripts/sub_$PART.sh
                        done
                    fi
                done
            done  
        done
    done
done
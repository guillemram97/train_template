#!/bin/sh
source scripts/cluster.sh
# Neptune tags 
TAGS="${TAGS:=debug}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:=yes}"

TASK_NAME="${TASK_NAME:=contmod}"
MODEL_NAME_OR_PATH="${BASE_MODEL:=google/gemma-2-27b-it}"
COST_EXT="${COST_EXT:=1}"
RETRAIN_FREQ="${RETRAIN_FREQ:=100}"
SOFT_LABELS="${SOFT_LABELS=0}"
N_INIT="${N_INIT:=100}"
CHECKPOINT="${CHECKPOINT:=-1}"
TEMPERATURE="${TEMPERATURE:=1.0}"
MAX_LEN="${MAX_LEN:=512}"
MAX_OUT_LEN="${MAX_OUT_LEN:=2}" # WE LIMIT FOR CLASSIFICATION
NUM_BEAMS="${NUM_BEAMS:=1}"
LR="${LR:=3e-5}"
BATCH="${BATCH:=16}"
BATCH_EVAL="${BATCH_EVAL:=16}"
EPOCHS="${EPOCHS:=100}" # Keep it fixed
WEIGHT_DECAY="${WEIGHT_DECAY:=1e-2}"

# Don't limit the number of examples and check how long it takes to execute a
# full run, maybe it takes less than a day and we're fine, if it's not then
# change the code and save checkpoint
TRAIN_SAMPLES="${TRAIN_SAMPLES:=10000}"
EVAL_SAMPLES="${EVAL_SAMPLES=100}"
TEST_SAMPLES="${TEST_SAMPLES=10000}"

# Early stopping
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:=2}"
EARLY_STOP="${EARLY_STOP:=20}"

R="${R:=16}" 
LORA_SCALING="${LORA_SCALING:=0.25}"


SEED="${SEED=0}"

python -m main \
  --model_name_or_path $BASE_MODEL \
  --task_name $TASK_NAME \
  --checkpoint $CHECKPOINT \
  --save_checkpoint $SAVE_CHECKPOINT \
  --temperature $TEMPERATURE \
  --max_length $MAX_LEN \
  --per_device_train_batch_size $BATCH \
  --per_device_eval_batch_size $BATCH_EVAL \
  --learning_rate $LR \
  --num_train_epochs $EPOCHS \
  --train_samples $TRAIN_SAMPLES \
  --eval_samples $EVAL_SAMPLES \
  --test_samples $TEST_SAMPLES \
  --eval_every_epochs $EVAL_EVERY_EPOCHS \
  --early_stop $EARLY_STOP \
  --r $R \
  --lora_scaling $LORA_SCALING \
  --num_beams $NUM_BEAMS \
  --weight_decay $WEIGHT_DECAY \
  --tags $TAGS \
  --max_out_length $MAX_OUT_LEN \
  --seed $SEED
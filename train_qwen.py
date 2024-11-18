import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, set_seed
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from enum import Enum
from datasets import DatasetDict, load_dataset, load_from_disk
import json
from multiprocessing import cpu_count
import os
import pdb
# os.environ["WANDB_DISABLED"] = "true"
# os.environ["WANDB_LOG_MODEL"] = "checkpoint"
#accelerate launch --config_file "fsdp_config.yaml" examples/trajectory_sft_mistral.py

set_seed(100)

# DEFAULT_CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
# DEFAULT_CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{ '<|' +  message['role'] + '|>' + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|assistant|>\n' }}{% endif %}{% endfor %}"


# DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

# DEFAULT_CHAT_TEMPLATE = "{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content'] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\n        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}\n    {%- endif %}\n    {%- if message['role'] == 'user' %}\n        {%- if loop.first and system_message is defined %}\n            {{- ' [INST] ' + system_message + '\\n\\n' + message['content'] + ' [/INST]' }}\n        {%- else %}\n            {{- ' [INST] ' + message['content'] + ' [/INST]' }}\n        {%- endif %}\n    {%- elif message['role'] == 'assistant' %}\n        {{- ' ' + message['content'] + eos_token}}\n    {%- else %}\n        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}\n    {%- endif %}\n{%- endfor %}\n"

# https://discuss.huggingface.co/t/mistralai-mistral-7b-instruct-v0-2-fine-tuning-prompt-format/68899/3

# output_dir = "qwen25-sft-aligned-actions-codex2-ef-gp-lr85"
output_dir = "qwen25-sft-scratchpad"

use_reentrant = False

# bnb_config = None
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_quant_storage=torch.bfloat16,
# )


training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=  "epoch", #"epoch",
        save_strategy = "steps",
        do_eval=True,
        eval_on_start=True,
        # eval_delay=0.0,
        # optim="adamw_8bit",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=1,
        log_level="info",
        save_steps=100,
        logging_steps=5,
        learning_rate=8e-5,
        #eval_steps=200,
        num_train_epochs=1,
        # max_steps=100,
        # warmup_steps=100,
        warmup_ratio=0.0,
        max_grad_norm=1.0,
        weight_decay=1e-4,
        lr_scheduler_type="cosine",#"linear",
        bf16= True,
        fsdp=[],
        gradient_checkpointing = True,
        # hub_private_repo=True,
        report_to="wandb"
)



use_4bit_quantization = True
use_8bit_quantization = False
bnb_config = None
quant_storage_dtype = None
if use_4bit_quantization:
    bnb_4bit_quant_type="nf4"
    # bnb_4bit_use_double_quant=True
    bnb_4bit_compute_dtype="bfloat16"
    bnb_4bit_quant_storage_dtype="bfloat16"
    use_nested_quant = False

    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    quant_storage_dtype = getattr(torch, bnb_4bit_quant_storage_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit_quantization,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
        bnb_4bit_quant_storage=quant_storage_dtype,
    )

    if compute_dtype == torch.float16 and use_4bit_quantization:
        major, _ = torch.cuda.get_device_capability()
        print(major)
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)
elif use_8bit_quantization: # Does not work
    bnb_config = BitsAndBytesConfig(load_in_8bit=use_8bit_quantization)



use_peft_lora = True
lora_r = 64
lora_alpha = 128
lora_dropout = 0.1
lora_target_modules="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj"
lora_target_modules = "all-linear"
chat_template_format = "chatml"

# Training Args
gradient_checkpointing = True


# dataset Args
data_args_packing = False

# model_name = "Qwen/Qwen2.5-Coder-7B"
model_name = "google/gemma-2-27b-it"
model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "Qwen/Qwen2.5-7B-Instruct"
max_seq_length =  400

dataset_text_field = None


# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token

# def preprocess(samples):
#     batch = []
#     for conversation in samples["history"]:
#         batch.append(tokenizer.apply_chat_template(conversation, tokenize=False))
#     return {"history": batch, "action":samples["action"]}

def preprocess(samples):
    batch = []
    actions = []
    for conversation, action in zip(samples["text"],samples['label']):
        conv = [{'role':'user', 'content':conversation}]
        x_sample = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        tok_example = tokenizer(x_sample, return_tensors="pt")
        if tok_example["input_ids"].size()[1]< 11400:
            batch.append(x_sample)
            actions.append(action)
    return {"history": batch, "action":actions}

# def preprocess(samples):
#     batch = []
#     actions = []
#     for conversation, i, action in zip(samples["history"],samples['iteration'],samples['action']):
#         if i==0:
#             batch.append(tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True))
#             actions.append(action)
#     return {"history": batch, "action":actions}

torch_dtype = torch.bfloat16
print(torch_dtype)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype=torch_dtype
)

peft_config = None
chat_template = None
if use_peft_lora:
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules.split(",")
        if lora_target_modules != "all-linear"
        else lora_target_modules,
    )


tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    # pad_token=special_tokens.pad_token.value,
    # bos_token=special_tokens.bos_token.value,
    # eos_token=special_tokens.eos_token.value,
    # additional_special_tokens=special_tokens.list(),
    # trust_remote_code=True
)


# set pad_token_id equal to the eos_token_id if not set
# if tokenizer.pad_token_id is None:
#   tokenizer.pad_token_id = tokenizer.unk_token_id

# tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
# make embedding resizing configurable?
# model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

model.config.pad_token_id = tokenizer.pad_token_id
# gradient ckpt
model.config.use_cache = not gradient_checkpointing
if gradient_checkpointing:
    gradient_checkpointing_kwargs = {"use_reentrant": use_reentrant}

# model.save_pretrained("hf_mistral_int_v2")
# tokenizer.save_pretrained("hf_mistral_int_v2")
# quit()
# datasets
raw_datasets = DatasetDict()
# for split in data_args.splits.split(","):
#     try:
#         # Try first if dataset on a Hub repo
#         dataset = load_dataset(data_args.dataset_name, split=split)
    # except DatasetGenerationError:
    #     # If not, check local dataset
    #     dataset = load_from_disk(os.path.join(data_args.dataset_name, split))

# raw_datasets = load_dataset("json", data_files={'train': "output_train.jsonl", 'test': "output_test.jsonl"})
# raw_datasets["train"] = load_dataset(dataset_name, split="train")
train_dataset = load_dataset("stanfordnlp/imdb", split="train")
test_dataset = load_dataset("stanfordnlp/imdb", split="test")

# Combine into a DatasetDict
raw_datasets = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})
# 

# raw_datasets["test"] = load_dataset(dataset_name, split="test")
# else:
#     raise ValueError(f"Split type {split} not recognized as one of test or train.")

raw_datasets = raw_datasets.map(
    preprocess,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
    desc="Applying chat template",
)

train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["test"]
print(f"Size of the train set: {len(train_dataset)}. Size of the validation set: {len(eval_dataset)}")
print(f"A sample of train dataset: {train_dataset[0]}")

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['history'])):
        text = f"###HISTORY{example['history'][i]}START###ACTION{example['action'][i]}{tokenizer.eos_token}"
        output_texts.append(text)
    return output_texts

instruction_template = "###HISTORY"
response_template = "START###ACTION"
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, 
                                           response_template=response_template, 
                                           tokenizer=tokenizer, mlm=False)




# print(training_args)
# quit()

# print(training_args)
# 14374, 41895


# trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    packing=data_args_packing,
    # dataset_kwargs={
    #     "append_concat_token": False,
    #     "add_special_tokens": False,
    # },
    dataset_text_field=dataset_text_field,
    max_seq_length=max_seq_length,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

trainer.accelerator.print(f"{trainer.model}")
if hasattr(trainer.model, "print_trainable_parameters"):
    trainer.model.print_trainable_parameters()

# train
checkpoint = None
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
trainer.train(resume_from_checkpoint=checkpoint)

# saving final model
if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
trainer.save_model()


# accelerate launch --config_file "examples/accelerate_configs/fsdp_qlora.yaml" examples/trajectory_sft_mistral_v2.py |& tee tmux_without-wom-r32.log
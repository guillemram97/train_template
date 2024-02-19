from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import math
import argparse
import pandas as pd
import numpy as np
import pdb
import torch

#POSSIBLE MODELS
# tiiuae/falcon-40b
# meta-llama/Llama-2-7b-hf
# mistralai/Mixtral-8x7B-Instruct-v0.1
# mistralai/Mistral-7B-Instruct-v0.2

def make_short_name(s):
    org = s.split('/')[0]
    s = s.split('/')[1]
    if org =='tiiuae':
        dim = s.split('-')[1]
    else:
        dim = s.split('-')[2]
    name = s.split('-')[0]
    return name+'-'+dim+'-4b'

parser = argparse.ArgumentParser()
parser.add_argument(
        "--task",
        type=str,
        help="The name of the task to train on.",
)
parser.add_argument(
        "--model",
        type=str,
        help="The name of the task to train on.",
)
args = parser.parse_args()
task = args.task
model_name = args.model
model_short_name = make_short_name(model_name)
classification = False

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True, return_dict_in_generate=True, output_scores=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# THIS VERSION DEALS WITH QA TASKS!
if task in ['wikifact', 'narrative_qa', 'natural_qa', 'babi_qa']:
    classification = False
    df = pd.read_csv('../data/'+task+'/dataset.csv')
    prompts = df['prompt']

else:
    classification = True
    df = pd.read_csv('../datasets/data/'+task+'/test_soft.csv')
    with open('../datasets/data/'+task+'/config.json') as f:
        config = json.load(f)
    BATCH_SIZE = 1
    prompts = []
    num_batches = math.ceil(len(df)/BATCH_SIZE)
    for idx in range(num_batches):
        if BATCH_SIZE == 1: 
            aux = df.iloc[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]['input']
            aux = aux.iloc[0]
            tmp = config['prompt_in'] + aux + config['prompt_out']
            prompts.append(tmp)
    tgt_classes = config['classes']
    class_tokens = {}
    for tgt_class in tgt_classes:
        if model_name.split('/')[0] == 'tiiuae':
            aux = tokenizer.encode(tgt_class)[0]
        else:
            aux = tokenizer.encode(tgt_class)[1] #For mixtral this should be [1]
        class_tokens[aux]=tgt_class


with open("output/"+task+"/"+model_short_name+"_answer.txt", "a") as answer_file:
    with open("output/"+task+"/"+model_short_name+".txt", "a") as tgt_file:
        for prompt in prompts:
            input = []
            input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
            input = input_ids
            if classification: max_new_tokens = 1
            else: max_new_tokens = 30
            # prob_classes = logits[:, -1, torch.tensor(list(class_tokens.keys()))]
            if model_name.split('/')[0] == 'meta-llama':
                with torch.no_grad(): output = model(input)
                logits = output[0][0][-1]
                with torch.no_grad(): output = model.generate(input, max_new_tokens=max_new_tokens, temperature=0.000001)
                aux_answer = tokenizer.decode(output[0])[len(prompt)+4:]

            elif model_name.split('/')[0] == 'mistralai':
                with torch.no_grad(): output = model.generate(input,  max_new_tokens=max_new_tokens,temperature=0.0) # change this depending on the q.
                logits = output[1][0][0]
                aux_answer = tokenizer.decode(output[0][0])[len(prompt)+4:]
            elif model_name.split('/')[0] == 'tiiuae':
                with torch.no_grad(): output = model.generate(input,  max_new_tokens=max_new_tokens,temperature=0.0)
                logits = output[1][0][0]
                aux_answer = tokenizer.decode(output[0][0])[len(prompt):]
            if classification:
                prob_classes = logits[torch.tensor(list(class_tokens.keys()))]
            else: prob_classes = logits
            prob_classes = prob_classes.tolist()

            if '\n\n' in aux_answer: aux_answer = aux_answer[:aux_answer.find('\n\n')]
            if '</' in aux_answer: aux_answer = aux_answer[:aux_answer.find('</')]
            
            answer_file.write(aux_answer+'\n')
            tgt_file.write(str(prob_classes)+'\n')
tgt_file.close()
answer_file.close()
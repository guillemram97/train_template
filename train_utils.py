import torch
import pdb
import os
import json
from argparse import Namespace
import evaluate
from task import get_task
from metrics import Metric
import numpy as np
from tqdm import tqdm
from torch.nn import CrossEntropyLoss, Softmax

from utils import set_seeds

from transformers import T5ForConditionalGeneration, BertModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, GPTNeoXForCausalLM, AutoModelForSequenceClassification


cross_entropy = CrossEntropyLoss()
softmax = Softmax()


def get_model(args, nclasses=0):
    if args.model_name_or_path in ['bert-base']:
        model = BertModel.from_pretrained( args.model_name_or_path,)
    elif args.model_name_or_path[1:] in ['5-base', '5-large']:
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path,)
    elif args.model_name_or_path in ['EleutherAI/pythia-1b-deduped']:
        model = GPTNeoXForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map = 'auto')
    return model


def load_optimizer(model, args):
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "alphas" not in n],
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    return optimizer


def finish_training(accelerator, model, eval_metric, args):
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )


# de moment nomes generar prob de un token
def soft_loss(logits, soft_labels, temperature=1):
    # logits es BS x LENGTH x VOCAB
    # soft labels seran probs de 3-4 tokens en classification tasks
    # en altres tasks seran altres tokens ?
    cross_batch = 0
    for idx, label in enumerate(soft_labels):
        logits[idx] = softmax(logits[idx])
        label = softmax(label / temperature)
        cross_batch += cross_entropy(logits[idx], label)
    return cross_batch / logits.shape[0]


def classification_loss(logits, labels):
    print('soft_loss')
    return

# de moment nomes generar prob de un token
def soft_loss_weighted(logits, soft_labels, temperature=1):
    # afegir weights
    cross_batch = 0
    for idx, label in enumerate(soft_labels):
        logits[idx] = softmax(logits[idx])
        label = softmax(label / temperature)
        factor = 1
        if label[1] > label[0]:
            factor = 10
        cross_batch += factor * cross_entropy(logits[idx], label)
    return cross_batch / logits.shape[0]


def train_epoch(
    model,
    train_dataloader,
    accelerator,
    lr_scheduler,
    optimizer,
    args,
    classification,
    dic_classes=None,
    ):

    model.train()
    set_seeds(args.seed)
    total_loss = 0

    losses = []

    freq = 100

    for step, batch in enumerate(train_dataloader):

        outputs = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.gold_hard,
        )
        if not classification:
            loss = ((outputs.logits - batch.gold_hard)**2).mean()
        else:
            loss = outputs.loss

        total_loss += loss.detach().float().item()
        losses.append(loss.detach().float().item())

        accelerator.backward(loss)

        if step % freq == 0:
            print(f"loss = {sum(losses) / len(losses)}")
            losses = []

        optimizer.step()
        lr_scheduler.step()

        optimizer.zero_grad()

    return total_loss


def evaluate_model(
    model, accelerator, eval_dataloader, metric, args, dic_classes
):
    model.eval()
    samples_seen = 0

    for step, batch in tqdm(enumerate(eval_dataloader)):
        with torch.no_grad():
            if model.config._name_or_path=='distilbert-base-uncased':
                if metric.soft == False:
                    predictions = torch.tensor(model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits).argmax(1) #aixo diferent
                else:
                    predictions = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits[:, 0].tolist()
                references = batch.gold_hard
                if torch.is_tensor(references):
                    references = [ref[0] for ref in references.tolist()]
       
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(eval_dataloader) - 1:
                predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                references = references[: len(eval_dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()

    if isinstance(eval_metric, dict):
        _, eval_metric = list(eval_metric.items())[0]
        # eval_metric = sum(eval_metric.values()) / len(eval_metric)

    return eval_metric


def save_model(accelerator, epoch, args):
    output_dir = f"epoch_{epoch}"
    if args.output_dir is not None:
        output_dir = os.path.join(args.output_dir, output_dir)
    accelerator.save_state(output_dir)


def get_hparams(args, task_name):
    PATH_PARAMS = "hparams/" + task_name + "/params.json"
    if os.path.exists(PATH_PARAMS):
        f = open(PATH_PARAMS)
        data = json.load(f)
        aux_args = vars(args)
        for hparam in data:
            aux_args[hparam] = data[hparam]
        args = Namespace(**aux_args)

    return args

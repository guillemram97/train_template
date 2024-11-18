import os
from transformers import DataCollatorForSeq2Seq, BertTokenizer, AutoTokenizer
from datasets import arrow_dataset
from torch.utils.data import DataLoader
import datasets
from torch.utils.data import Subset
import torch
from multiprocessing import cpu_count
from functools import partial
import pandas as pd
import ast
import json
import pdb


class Task:
    def __init__(self, task_name, tokenizer):
        self.task_name = task_name
        DATA_DIR = os.getenv("DATA_PATH")
        self.path = os.path.join(DATA_DIR, task_name)
        self.tokenizer = tokenizer
        self.load_config()
        self.load_data()

    def load_config(self):
        with open(os.path.join(self.path, "config.json")) as f:
            config = json.load(f)
            self.classes = config["classes"]
            self.system = config["system"]

    def load_data(self):
        data = {}
        df = pd.read_csv(os.path.join(self.path, "data.csv"), nrows=10000)
        text = list(df["prompt"].values.astype(str))
        label = list(df["output"].values.astype(str))
        
        data = arrow_dataset.Dataset.from_dict(
            {
                "text": text,
                "label": label,
            }
        )
        data_train = data[1250:]
        data_eval = data[1000:1250]
        data_test = data[:1000]
        self.raw_data = {}
        self.raw_data['train'] = datasets.DatasetDict(data_train)
        self.raw_data['eval'] = datasets.DatasetDict(data_eval)
        self.raw_data['test'] = datasets.DatasetDict(data_test)

    def load_classes(self):
        self.classes_dict = {}
        for idx, class_name in enumerate(self.classes):
            target = self.tokenizer.encode(class_name, add_special_tokens=False)#[0] # I should check this for Mixtral as well!
            self.classes_dict[class_name] = target
        return
        

    def preprocess(self, accelerator, args, model=None):
        def process_data_to_model_inputs(is_eval: bool, batch):
            out = {}
            # Tokenizer will automatically set [BOS] <text> [EOS]
            #out["input_ids"] = self.tokenizer(
            #    batch["text"],
            #    padding=False,
            #    max_length=args.max_length,
            #    truncation=True,
            #).input_ids
            out["text"] = batch["text"]
            out["label"] = [list(self.classes_dict.keys()).index(element) for element in batch['label']]
            #out["label"] = self.tokenizer(
            #    batch["label"],
            #    padding=False,
            #    max_length=args.max_length,
            #    truncation=True,
            #).input_ids#list(self.classes_dict.keys()).index(element) for element in batch['label']]
            return out

        def collate_for_eval(default_collate, batch):
            text = [{"input_ids": x["input_ids"]} for x in batch]
            out = default_collate(text)
            out["label"] = [x["label"] for x in batch]
            return out


        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=model, padding="longest"
        )
        eval_collator = partial(collate_for_eval, data_collator)

        processed_data = {}

        for split in ['train', 'eval', 'test']:
            max_samples = getattr(args, f"{split}_samples")
            tmp_list = []
            for input, output in zip(self.raw_data[split]['text'], self.raw_data[split]['label']): tmp_list.append({"text": input, "label": output})
            self.raw_data[split] = arrow_dataset.Dataset.from_list(
                tmp_list
            )
            processed_data[split] = self.raw_data[split].map(
                partial(process_data_to_model_inputs, split in ["test"]),
                batched=True,
                batch_size=args.per_device_eval_batch_size,
                remove_columns=self.raw_data[split].column_names,
            )


        train_dataloader = DataLoader(
            processed_data["train"],
            shuffle=False,
            collate_fn=data_collator,
            batch_size=args.per_device_train_batch_size,
        )

        test_dataloader = DataLoader(
            processed_data["test"],
            collate_fn=eval_collator,
            batch_size=args.per_device_eval_batch_size,
        )

        eval_dataloader = DataLoader(
            processed_data["eval"],
            collate_fn=eval_collator,
            batch_size=args.per_device_eval_batch_size,
        )
        
        train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
            train_dataloader,
            eval_dataloader,
            test_dataloader,
        )

        self.data = {
            "train_dataloader": train_dataloader,
            "eval_dataloader": eval_dataloader,
            "test_dataloader": test_dataloader,
        }
        return


def random_subset(dataset, max_samples: int, seed: int = 42):
    if max_samples >= len(dataset) or max_samples == -1:
        return dataset

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=generator)
    return Subset(dataset, perm[:max_samples].tolist())


def get_task(accelerator, args, model=None):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, model_max_length=args.max_length, use_auth_token=True
    )

    # load config, data, and preprocess
    task = Task(args.task_name, tokenizer)
    task.load_classes()
    task.preprocess(accelerator, args, model=None)
    return task


def apply_chat_template(sys_prompt, prompt):
    messages = [
        {"role": "user", "content": sys_prompt + "\n\n" + prompt},
    ]
    return messages
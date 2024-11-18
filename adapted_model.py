from adapters.load_mcl import ModularMixin
from train_utils import (
    load_optimizer,
    evaluate_model,
    train_epoch,
    get_hparams,
    get_model,
)
from peft import LoraConfig
import torch.nn as nn
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate import dispatch_model, infer_auto_device_map
from transformers import (
    get_scheduler,
    TrainingArguments,
)
from utils import (
    setup_basics,
    EarlyStopper,
    neptune_log,
    set_seeds,
)
import copy
from task import (
    get_task,
)
from metrics import Metric
import pdb
import time
import numpy as np

logger = get_logger(__name__)

LOG_TRAIN = True


class AdaptedModel:
    def __init__(self, args, task, run, accelerator):
        self.cache = []
        self.task_name = args.task_name
        self.seed = args.seed
        self.args = get_hparams(args, self.task_name)
        self.test = task.data["test_dataloader"]
        self.run = run
        self.seed = args.seed
        self.accelerator = accelerator
        self.iteration = 0
        self.save_checkpoint = args.save_checkpoint
        self.test_scores_gold = [0, 0]
        self.suffixes = [""]
        self.dic_classes = list(task.classes_dict.values())
        self.n_classes = len(list(task.classes_dict.keys()))
        
        self.init_model()
        # EVAL METRIC
        self.metric = Metric(self.args, classification=True)
        # TEST METRIC
        self.metric_test = Metric(self.args, classification=True)

    def init_model(self):
        set_seeds(self.seed)
        model = get_model(self.args, self.n_classes)
        #device_map = infer_auto_device_map(model, max_memory = {0: "23GB", 1: "23GB", 2: "23GB", 3: "23GB", 4: "23GB", 5: "23GB", 6: "23GB", 7: "23GB"})
        #self.model = ModularMixin(
        #    model,
        #    freeze=True,
        #    ac_kwargs={
        #        "r": self.args.r,
        #        "lora_scaling": self.args.lora_scaling,
        #        "seed": self.seed,
        #    },
        #)

        use_peft_lora = True
        lora_r = 64
        lora_alpha = 128
        lora_dropout = 0.1
        lora_target_modules="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj"
        chat_template_format = "chatml"
        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_target_modules.split(","),
        )
        self.model = model

        self.peft_config = peft_config
        return

    def init_checkpoint(self, PATH):
        self.model.load_state_dict(torch.load(PATH))
        #self.model.cuda()
        #we need to do something about device_map!
        #model = dispatch_model(model, device_map=device_map)
        return

    def evaluate(self):
        self.metric_test.reset()
        test_metric = evaluate_model(
            model=self.model,
            accelerator=self.accelerator,
            eval_dataloader=self.test,
            metric=self.metric_test,
            args=self.args,
            dic_classes=self.dic_classes,
        )

        if self.run is not None:
            if self.is_classification:
                stats = {
                    "test_gold_acc": test_metric[0],
                    "test_gold_f1": test_metric[1],
                    "data amount": self.data_amount,
                }
            else:
                stats = {
                        "test_gold_L2": test_metric[0],
                        "data amount": self.data_amount,
                }    
            neptune_log(
                run=self.run,
                pref=f"test/",
                stats=stats,
                epoch=self.iteration,
            )


    def train(self, train_dataloader, eval_dataloader, tokenizer):
        from datasets import load_dataset
        from transformers import DataCollatorWithPadding


        def preprocess_function(examples, label_mapping):
            inputs = examples['text']
            targets = examples['label']
            model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=128)
            targets = [label_mapping[label] for label in examples['label']]
            with tokenizer.as_target_tokenizer():

                labels = tokenizer(targets, padding="max_length", truncation=True, max_length=128)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        dataset = load_dataset("stanfordnlp/imdb", split="train")

        model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", device_map='auto')
        tokenizer.pad_token = tokenizer.eos_token
        #data_collator = CustomDataCollator(tokenizer)
        data_collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, response_template='')
        training_args = SFTConfig(output_dir="/tmp")

        # borrar aixo!
        self.dic_classes = {0: 'Safe', 1: 'Unsafe'}
        tokenized_dataset = dataset.map(lambda examples: preprocess_function(examples, self.dic_classes), batched=True, remove_columns=dataset.column_names)
        trainer = SFTTrainer(
            model,
            train_dataset=tokenized_dataset,
            args=training_args,
            data_collator=data_collator,
        )
        trainer.train()






        def formatting_prompts_func(example):
            output_texts = []
            for i in range(len(example['itest'])):
                #text = f"###HISTORY{example['history'][i]}START###ACTION{example['action'][i]}{tokenizer.eos_token}"
                output_texts.append(example['text'][i])
            return output_texts
        
        t = time.time()
        self.early_stopper = EarlyStopper(self.args.early_stop)
        self.iteration += 1
        if self.seed is not None:
            set_seed(self.args.seed)

        self.metric.reset()

        # for every retraining, we train from scratch
        #self.init_model()

        logger.info(f"  Running task {self.task_name}")
        logger.info(f"  Num examples = {len(train_dataloader.dataset)}")

        self.data_amount = len(train_dataloader.dataset) + len(eval_dataloader.dataset)
        # Re-initialise lr_scheduler + optimized
        optimizer = load_optimizer(self.model, self.args)

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=int(
                self.args.warmup
                * self.args.num_train_epochs
                * len(
                    train_dataloader.dataset
                )  # aixo esta be? no hauria de ser len(train_dataloader.dataset)?
            ),
            num_training_steps=self.args.num_train_epochs
            * len(
                train_dataloader.dataset
            ),  # aixo esta be? no hauria de ser len(train_dataloader.dataset)?
        )


        #optimizer, lr_scheduler = self.accelerator.prepare(
        #    optimizer, lr_scheduler
        #) #self.model, 
        tokenizer.padding_side = 'right'
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        training_args = TrainingArguments(
                output_dir='checkpoints',
                evaluation_strategy=  "epoch", #"epoch",
                save_strategy = "steps",
                do_eval=True,
                eval_on_start=False,
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
                report_to="neptune"
        )
        data_args_packing = False
        dataset_text_field = None
        max_new_tokens=1
        instruction_template = ''
        response_template = ''
        from datasets import load_dataset
        dataset = load_dataset("stanfordnlp/imdb", split="train")
        dataset_eval = load_dataset("stanfordnlp/imdb", split="test")
        instruction_template = "### Human:"
        response_template = "### Assistant:"

        #collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, 
        #                                   response_template=response_template, 
        #                                   tokenizer=tokenizer, mlm=False)
        
        collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=dataset, #train_dataloader.dataset, 
            eval_dataset=dataset_eval, #eval_dataloader.dataset,
            peft_config=self.peft_config,
            packing=data_args_packing,#dataset_text_field=dataset_text_field,
            max_seq_length=1000, #max_new_tokens=max_new_tokens, #data_collator=collator, #formatting_func=formatting_prompts_func,
        )
        print('ENTRENANTTTTTTTTTTT')
        trainer.train()


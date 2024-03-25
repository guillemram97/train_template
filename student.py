from adapters.load_mcl import ModularMixin
from train_utils import (
    load_optimizer,
    evaluate_model,
    train_epoch,
    get_hparams,
    get_model,
)
import torch.nn as nn
import torch
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    get_scheduler,
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


class student:
    def __init__(self, args, task, run, accelerator):
        self.cache = []
        self.task_name = args.task_name
        self.seed = args.seed
        self.target = args.target
        self.args = get_hparams(args, self.task_name)
        self.test = task.data["test_dataloader"]
        self.run = run
        self.seed = args.seed
        self.accelerator = accelerator
        self.iteration = 0
        self.save_checkpoint = args.save_checkpoint
        self.soft_labels = args.soft_labels
        self.test_scores_gold = [0, 0]
        self.test_scores_llm = [0, 0]
        self.suffixes = [""]
        self.is_classification = task.is_classification
        if task.is_classification:
            self.dic_classes = list(task.classes_dict_gold.values())
            self.n_classes = len(list(task.classes_dict_gold.keys()))
        else:
            self.dic_classes = None
            self.n_classes = 1
            self.soft_labels = True
        
        self.init_model()

        # need to revise these two!!!!
        self.metric = Metric(self.args, soft=self.soft_labels, classification=task.is_classification)
        self.metric_test = Metric(self.args, soft=self.soft_labels, classification=task.is_classification)

    def init_model(self):
        set_seeds(self.seed)
        model = get_model(self.args, self.n_classes)
        model.cuda()
        self.model = model
        return

    def init_checkpoint(self, PATH):
        self.model.load_state_dict(torch.load(PATH))
        self.model.cuda()
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
            target="gold",
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


    def train(self, train_dataloader, eval_dataloader):
        torch.cuda.empty_cache()
        t = time.time()
        self.early_stopper = EarlyStopper(self.args.early_stop, self.is_classification)
        self.iteration += 1
        if self.seed is not None:
            set_seed(self.args.seed)

        self.metric.reset()

        # for every retraining, we train from scratch
        self.init_model()

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

        # Move to the device
        self.model, optimizer, lr_scheduler = self.accelerator.prepare(
            self.model, optimizer, lr_scheduler
        )

        for epoch in range(0, self.args.num_train_epochs):
            total_loss = train_epoch(
                model=self.model,
                train_dataloader=train_dataloader,
                accelerator=self.accelerator,
                lr_scheduler=lr_scheduler,
                optimizer=optimizer,
                args=self.args,
                classification=self.is_classification,
                dic_classes=self.dic_classes,
            )

            if (
                epoch % self.args.eval_every_epochs == 0
                or epoch == self.args.num_train_epochs - 1
            ):
                eval_metrics = evaluate_model(
                    model=self.model,
                    accelerator=self.accelerator,
                    eval_dataloader=eval_dataloader,
                    metric=self.metric,
                    args=self.args,
                    dic_classes=self.dic_classes,
                    target=self.target,
                )
                self.model.cpu()

                # early stopper requires an increasingly increasing metric
                # we can use epochs instead of eval_metrics[0]
                self.early_stopper.update(eval_metrics[-1], self.model)
                self.model.cuda()

                log_msg = f"Epoch {epoch} -----> Average_Train_loss: {total_loss / len(train_dataloader.dataset)} ===== Eval_metric: {eval_metrics[0]}"
                logger.info(log_msg)

                if self.run is not None and LOG_TRAIN:
                    if self.is_classification:
                        self.run[f"{self.iteration}-eval-acc"].log(eval_metrics[0], step=epoch)
                        self.run[f"{self.iteration}-eval-f1"].log(eval_metrics[1], step=epoch)
                    else:
                        self.run[f"{self.iteration}-eval-L2"].log(eval_metrics[0], step=epoch)
            # log metrics are desactivated
            if self.run is not None and LOG_TRAIN:
                stats = {
                    "loss": total_loss / len(train_dataloader.dataset),
                    "main_lr": optimizer.param_groups[0]["lr"],
                }

                neptune_log(
                    run=self.run,
                    pref=f"{self.iteration}-train/",
                    stats=stats,
                    epoch=epoch,
                 )

            if self.early_stopper.should_finish():
                break

        elapsed = time.time() - t
        print(elapsed)
        # copying from a cpu
        self.model.cpu()
        self.model = copy.deepcopy(self.early_stopper.get_best())
        self.model = self.early_stopper.get_best().cuda()
        del self.early_stopper.best_model
        # self.model.cuda()

        self.evaluate()
        if self.save_checkpoint != "no":
            PATH_DEST = (
                "/rds/user/cs-rami1/hpc-work/checkpoints/"
                 + self.args.model_name_or_path.split("/")[-1]
                + "/"
                + self.task_name
                + "/"
                + str(self.seed)
                + "_"
                + str(len(train_dataloader.dataset) + len(eval_dataloader.dataset))
                + ".pt"
            )
            torch.save(self.model.state_dict(), PATH_DEST)


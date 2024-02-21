from utils import (
    parse_args,
    setup_basics,
    neptune_log,
    set_seeds,
)
from utils.online_logs import (
    log_test,
    log_final,
)
import numpy as np
from metrics import Metric
from student import student
from cache import cache_store
from accelerate import Accelerator
from accelerate.logging import get_logger
from task import (
    get_task,
    make_datacollator,
)
import pdb
import copy
import gc

logger = get_logger(__name__)


def main():
    args = parse_args()
    accelerator = Accelerator()
    run = setup_basics(accelerator, logger, args)

    # Pre-Logging
    run["args"] = vars(args)
    set_seeds(args.seed)

    task = get_task(
        accelerator=accelerator,
        args=args,
        model=None,
    )
    if not task.is_classification:
        args.is_classification = False
    else:
        args.soft_labels = (
            False  # MIRAR AIXO, ESTAVA A TRUE
        )
    online_dataloader = task.data["online_dataloader"]
    st = student(args, task, run, accelerator)
    metric = Metric(args, soft=args.soft_labels, online=True)
    cache = cache_store(args)
    # Initialize student model
    # If we put a checkpoint, we load the model and we skip the first $checkpoint steps
    if args.checkpoint != "-1":
        PATH = "checkpoints/" + args.task_name + "/" + str(args.checkpoint) + ".pt"
        st.init_checkpoint(PATH)

    stop_retraining = False
    send_update = False

    for step, sample in enumerate(online_dataloader):
        # IF WE HAVE A CHECKPOINT, WE SKIP N_INIT STEPS
        if args.checkpoint == "-1" or step >= args.n_init:
            gc.collect()
            cache.save_cache(sample)

            if step + 1 and (step + 1) % args.retrain_freq == 0 and not stop_retraining:
                set_seeds(args.seed)
                cache_tmp = cache.retrieve_cache()
                train_dataloader, eval_dataloader = make_datacollator(
                    args, task.tokenizer, cache_tmp
                )
                train_dataloader, eval_dataloader = accelerator.prepare(
                    train_dataloader, eval_dataloader
                )
                st.train(train_dataloader, eval_dataloader)

                del train_dataloader, eval_dataloader

    if run is not None:
        run.stop()


if __name__ == "__main__":
    main()

from utils import (
    parse_args,
    setup_basics,
    neptune_log,
    set_seeds,
)
from accelerate.logging import get_logger

from utils.online_logs import (
    log_test,
    log_final,
)
import numpy as np
from metrics import Metric
from adapted_model import AdaptedModel
from accelerate import Accelerator
from task import (
    get_task,
)
import pdb
import copy
import gc

#logger = get_logger(__name__)
logger = get_logger(__name__, log_level="DEBUG")


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

    train_dataloader = task.data['train_dataloader']
    eval_dataloader = task.data['test_dataloader']
    st = AdaptedModel(args, task, run, accelerator)
    st.train(train_dataloader, eval_dataloader, task.tokenizer)

    if run is not None: run.stop()


if __name__ == "__main__":
    main()

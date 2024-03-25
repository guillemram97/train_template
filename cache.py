import copy
import torch



class cache_store:
    def __init__(self, args):
        self.cache = {}
        self.args = args
        return
    def save_cache(self, input):
        if not "input_ids" in self.cache:
            self.cache["input_ids"] = [torch.flatten(input.input_ids).tolist()]
            self.cache["gold_hard"] = [torch.flatten(input.gold_hard).tolist()]
            return
        self.cache["input_ids"].append(torch.flatten(input.input_ids).tolist())
        self.cache["gold_hard"].append(torch.flatten(input.gold_hard).tolist())

    def retrieve_cache(self):
        return self.cache

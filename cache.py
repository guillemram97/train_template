import copy
import torch



class cache_store:
    def __init__(self, args):
        self.cache = {}
        self.args = args
        return
    def save_cache(self, input):
        aux = copy.deepcopy(torch.flatten(input.llm_soft).tolist())
        aux.sort()
        aux = aux[-1] - aux[-2]
        if not "input_ids" in self.cache:
            self.cache["input_ids"] = [torch.flatten(input.input_ids).tolist()]
            self.cache["gold_hard"] = [torch.flatten(input.gold_hard).tolist()]
            #if self.task.is_classification:
            #    self.cache["gold_soft"] = [torch.flatten(input.gold_soft).tolist()]
            #    self.cache["llm_soft"] = [torch.flatten(input.llm_soft).tolist()]
            #self.cache["llm_hard"] = [torch.flatten(input.llm_hard).tolist()]
            return
        self.cache["input_ids"].append(torch.flatten(input.input_ids).tolist())
        self.cache["gold_hard"].append(torch.flatten(input.gold_hard).tolist())
        #if self.task.is_classification:
        #    self.cache["gold_soft"].append(torch.flatten(input.gold_soft).tolist())
        #    self.cache["llm_soft"].append(torch.flatten(input.llm_soft).tolist())
        #self.cache["llm_hard"].append(torch.flatten(input.llm_hard).tolist())

    def retrieve_cache(self):
        return self.cache

import numpy as np
import string
import re
import torch
from collections import Counter
from sklearn.metrics import matthews_corrcoef, f1_score
from scipy.stats import pearsonr
from rouge import Rouge
from torch.nn import CrossEntropyLoss, Softmax
import pdb
from transformers import T5Tokenizer, BertTokenizer, AutoTokenizer

cross_entropy = CrossEntropyLoss()
softmax = Softmax()


class Metric:
    """
    It's used only for dev / test evaluation
    Not during training
    """

    def __init__(self, args, classification=True):
        self.task_name = args.task_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, model_max_length=args.max_length
        )
        self.predictions = []
        self.references = []
        self.args = args
        self.classification = classification

    def reset(self):
        self.predictions = []
        self.references = []

    def add_batch(self, predictions, references):
        self.predictions = self.predictions + predictions
        self.references = self.references + references

    def compute(self):
        if self.classification:
            metrics = evaluate_classification(
                self.predictions, self.references
            )
        else:
            raise NotImplementedError
        self.reset()
        return metrics


def evaluate_classification(predictions, data):

    pdb.set_trace()

    acc = 0
    nll = 0
    acc_vec = []
    bt_vec = []
    deferral_score = 0
    for idx, pred in enumerate(predictions):
        acc += 1*(pred==data[idx])
        nll -= 0
        acc_vec.append(pred==data[idx])
        bt_vec.append()
    args = np.argsort(np.array(bt_vec))
    acc_vec = np.array(acc_vec[args])
    for idx, acc_tmp in enumerate(acc_vec): deferral_score += acc_tmp*(idx+1) 
    deferral_score = deferral_score/(len(acc_vec)**2)
    acc = acc / len(data)
    f1 = f1_score(predictions, data, average='macro')
    
    metrics = {'acc': acc, 'f1': f1, 'nll': nll, 'deferral': deferral_score, }
    
    return metrics



def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_trivial_white_space(text):
        while len(text) and text[0] == " ":
            text = text[1:]
        while len(text) and text[-1] == " ":
            text = text[:-1]
        return text

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(
        remove_trivial_white_space(remove_articles(remove_punc(lower(s))))
    )

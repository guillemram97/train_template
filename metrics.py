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

METRICS = {
    "rt-polarity": ["ACC"],
    "hatexplain_nt": ["Classification-F1"],
    "hatexplain": ["hatexplain"],
    "isear": ["Classification-F1"],
    "trec": ["Classification-F1"],
    "openbook": ["ACC"],
    "ag_news": ["ACC"],
    "cr": ["ACC"],
    "sst2": ["ACC"],
    "isear_mistral": ["ACC"],
    "rt-polarity_mistral": ["ACC"],
    "openbook_mistral": ["ACC"],
    "fever_mistral": ["ACC"],
    "isear_llama": ["ACC"],
    "rt-polarity_llama": ["ACC"],
    "openbook_llama": ["ACC"],
    "fever_llama": ["ACC"],
    "isear_llama_hard": ["ACC"],
    "rt-polarity_llama_hard": ["ACC"],
    "openbook_llama_hard": ["ACC"],
    "fever_llama_hard": ["ACC"],
    "openbook_nf": ["ACC"],
    "quoref": ["QA-F1"],
    "implicatures": ["ACC"],
    "fever": ["ACC"],
    "fever_explore": ["Classification-F1"],
    "mmlu-human": ["Classification-F1"],
    "mmlu-ss": ["Classification-F1"],
    "qa_wikidata": ["QA-F1"],
}


class Metric:
    """
    It's used only for dev / test evaluation
    Not during training
    """

    def __init__(self, args, soft=False, classification=False):
        self.task_name = args.task_name
        #self.tokenizer = BertTokenizer.from_pretrained(
        #    args.model_name_or_path, model_max_length=args.max_length
        #)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, model_max_length=args.max_length
        )
        self.predictions = []
        self.references = []
        self.online = False
        self.args = args
        self.soft = soft
        self.classification = classification

    def reset(self):
        self.predictions = []
        self.references = []

    def add_batch(self, predictions, references):
        if not self.soft:
            if type(predictions[0]) != str:
                predictions = self.tokenizer.batch_decode(
                    predictions,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            if type(references[0]) != str:
                references = self.tokenizer.batch_decode(
                    references,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
        self.predictions = self.predictions + predictions
        self.references = self.references + references

    def compute(self):
        if self.classification:
            metrics = evaluate_classification(
                self.predictions, self.references
            )

        elif self.soft:
            metrics = evaluate_soft_regression(
                self.predictions, self.references
            )
        else:
            metrics = evaluate_hard(self.predictions, self.references, self.task_name)
        self.reset()
        return metrics

def evaluate_soft_regression(predictions, data):
    diff = 0
    for idx, pred in enumerate(predictions):
        diff += (pred-data[0])**2
    return [diff/(idx+1)]

def evaluate_soft(predictions, data, temperature=1):
    cross_entropy_score = []
    accuracy = []
    # new_predictions = []
    # new_data = []
    for idx, pred in enumerate(predictions):
        predictions[idx] = softmax(torch.tensor(pred).cuda().float())
        data[idx] = softmax(torch.tensor(data[idx]).cuda().float() / temperature)
        cross_entropy_score.append(cross_entropy(predictions[idx], data[idx]))
        predictions[idx].argmax() == data[idx].argmax()
        # new_predictions.append(predictions[idx].argmax().cpu())
        # new_data.append(data[idx].argmax().cpu())
        accuracy.append(1 * (predictions[idx].argmax() == data[idx].argmax()).tolist())
    # if len(new_data) > 1: pdb.set_trace()
    # sum(accuracy) / len(data)
    # f1_score(new_data, new_predictions, average="macro")
    return [sum(accuracy) / len(data), sum(cross_entropy_score) / len(data)]


def evaluate_classification(predictions, data):
    acc = 0
    for idx, pred in enumerate(predictions):
        acc += 1*(pred==data[idx])
    acc = acc / len(data)
    return [acc, f1_score(predictions, data, average='macro')]


def evaluate_hard(predictions, data, task):
    metrics = METRICS[task]
    tmp_metrics = []
    assert len(predictions) == len(data)

    if "ACC" in metrics:
        accs = []
        for prediction, dp in zip(predictions, data):
            accs.append(get_accruacy_over_list(prediction, dp))
        tmp_metrics.append(np.mean(accs))
    if "hatexplain" in metrics:
        accs = []
        for prediction, dp in zip(predictions, data):
            accs.append(get_accuracy_hatexplain(prediction, dp))
        tmp_metrics.append(np.mean(accs))
    if "QA-F1" in metrics:
        f1s = []
        for prediction, dp in zip(predictions, data):
            f1s.append(get_f1_over_list(prediction, dp))
        tmp_metrics.append(np.mean(f1s))
    if "Classification-F1" in metrics:
        if isinstance(data[0], list):
            data = [dat[0] for dat in data]
        return tmp_metrics.append(f1_score(data, predictions, average="macro"))
    return tmp_metrics


def qa_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def accuracy(prediction, ground_truth):
    return prediction.lower() == ground_truth.lower()


def get_accuracy_hatexplain(prediction, groundtruth):
    prediction = prediction.split(" ")
    groundtruth = groundtruth.split(" ")
    precision = 0
    f1 = None
    if len(prediction[1:]):
        for pred in prediction[1:]:
            precision += get_accruacy_over_list(pred, groundtruth[1:])
        precision = precision / len(prediction[1:])
        f1 = precision
    recall = 0
    if len(groundtruth[1:]):
        for truth in groundtruth[1:]:
            recall += get_accruacy_over_list(truth, prediction[1:])
        recall = recall / len(groundtruth[1:])
        if f1 is None:
            f1 = recall
        else:
            f1 = 0.5 * precision + 0.5 * recall
    if f1 is None:
        return accuracy(prediction[0], groundtruth[0])
    return 0.5 * accuracy(prediction[0], groundtruth[0]) + 0.5 * f1


def get_accruacy_over_list(prediction, groundtruth):
    if isinstance(groundtruth, list):
        if len(groundtruth) == 0:
            return 0
        return np.max([accuracy(prediction, gt) for gt in groundtruth])
    return accuracy(prediction, groundtruth)


def get_f1_over_list(prediction, groundtruth):
    if isinstance(groundtruth, list):
        if len(groundtruth) == 0:
            return 0
        return np.max([qa_f1_score(prediction, gt) for gt in groundtruth])
    return qa_f1_score(prediction, groundtruth)


def get_exact_match_over_list(prediction, groundtruth):
    if isinstance(groundtruth, list):
        if len(groundtruth) == 0:
            return 0
        return np.max([get_exact_match_over_list(prediction, gt) for gt in groundtruth])
    return normalize_answer(prediction) == normalize_answer(groundtruth)


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

import os
import inspect

import logging
import json
import pickle

# import time
import random
from typing import List, Dict

import numpy as np

# import evaluate
import torch
# import torch.nn.functional as F
from accelerate.utils import set_seed
from torch import nn
from torch.utils import data
from transformers import default_data_collator
from transformers import AutoTokenizer, PreTrainedTokenizer

# tested with captum 0.7.0

from ferret import LIMEExplainer
from ferret import SHAPExplainer
from ferret import GradientExplainer
from ferret import IntegratedGradientExplainer
from ferret import Benchmark

# from sklearn.metrics import precision_recall_fscore_support, classification_report

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# check if cuda or mps is available
if torch.cuda.is_available():
    DEVICE = 'cuda'
# elif torch.backends.mps.is_available():
#     DEVICE = 'mps'
else:
    DEVICE = 'cpu'

# print(f"Using device: {DEVICE}")


def function_accepts_argument(func, arg_name):
    signature = inspect.signature(func)
    return arg_name in signature.parameters




def ferret_interpret_model(model, tokenizer, dataset, 
                           label_key: str='label',) -> List:

    ig = IntegratedGradientExplainer(model, tokenizer, multiply_by_inputs=True)
    g = GradientExplainer(model, tokenizer, multiply_by_inputs=True)
    l = LIMEExplainer(model, tokenizer)
    s = SHAPExplainer(model, tokenizer)

    bench = Benchmark(model, tokenizer,explainers=[ig, g, l, s])
    num_labels = model.config.num_labels

    eval_results = []
    for instance in dataset:
        text = tokenizer.decode(instance["input_ids"], skip_special_tokens=False)
        # print('Text:', text, 'Label:', instance[label_key], num_labels)
        instance_result = {
            "text": text,
            label_key: instance[label_key],
            "correct_results": [],
            "incorrect_results": []
        }
        # get model prediction
        # print("Instance:", instance)
        instance = default_data_collator([instance])
        inp = {
            "input_ids": instance["input_ids"].to(device=DEVICE),
            "attention_mask": instance["attention_mask"].to(device=DEVICE),
        }
        if "token_type_ids" in instance:
            inp["token_type_ids"] = instance["token_type_ids"].to(device=DEVICE)

        logits = model(**inp).logits
        # labels = instance["labels"].to(device=DEVICE)
        prediction = logits.argmax(dim=-1)
        prediction = prediction.detach().cpu()

        class_idx = prediction.item()
        result = bench.explain(text, class_idx)
        # print("Result:", result)

        evals_result = bench.evaluate_explanations(result, class_idx)
        if random.random() < 0.5:
            print("Bench results:", evals_result)

        if instance_result[label_key] == class_idx:
            instance_result["correct_results"].append(evals_result)
        else:
            instance_result["incorrect_results"].append(evals_result)
        eval_results.append(instance_result)
    
    return eval_results


def seralize_attributions(attributions: List[Dict]):
    # check if the attributions is a list of dictionaries and each element is serializable
    for attr in attributions:
        if not isinstance(attr, dict):
            raise ValueError("Attributions should be a list of dictionaries")
        for key, value in attr.items():
            if not isinstance(value, (int, float, str, list)):
                raise ValueError("Attributions should be serializable")
    return True



def save_attributions(attributions: List[Dict], path: str):
    # check serializability of the attributions
    if seralize_attributions(attributions):
        with open(path, "w") as f:
            json.dump(attributions, f)


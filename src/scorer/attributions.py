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

from peft import TaskType

# tested with captum 0.7.0

from ferret import LIMEExplainer
from ferret import SHAPExplainer
from ferret import GradientExplainer
from ferret import IntegratedGradientExplainer
from ferret import Benchmark

from src.scorer.utils_moodel import load_lora_model

# from sklearn.metrics import precision_recall_fscore_support, classification_report

from src.scorer.utils_cli import parse_args

from src.scorer.pipeline import construct_model
from src.scorer.pipeline import get_csv_dataset

from src.scorer.utils_torch import get_nested_attr

# check if cuda or mps is available
if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

# print(f"Using device: {DEVICE}")

def function_accepts_argument(func, arg_name):
    signature = inspect.signature(func)
    return arg_name in signature.parameters



def attr_parse_args():
    attr_parser = parse_args("Run and Evaluate a model attribution on a custom CSV dataset.")

    # attr_parser.add_argument("--num_labels", type=int, help="Number of labels in the dataset")

    # Add attribution arguments
    attr_parser.add_argument("--attr_method", type=str, default="lig", 
                             help="Attribution method to use")
    attr_parser.add_argument("--embeddings", type=str, default="embeddings", 
                             help="embedding layer name")
    attr_parser.add_argument("--attr_target_layer", type=str, default="embeddings", 
                             help="Target layer for attribution")

    # Load LoRA model arguments
    attr_parser.add_argument("--load_lora", action="store_true", 
                             help="Load LoRA model")
    attr_parser.set_defaults(load_lora=False)

    attr_parser.add_argument("--target_modules", type=str, nargs="+", default=None, 
                             help="Target modules to apply LoRA")
    attr_parser.add_argument("--num_samples", type=int, default=None, 
                             help="Number of samples to use for attribution methods that require sampling")
    
    attr_parser.add_argument("--output_attr_dir", type=str, default=None,)

    args = attr_parser.parse_args()

    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset_name)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    if args.output_attr_dir is not None:
        os.makedirs(args.output_attr_dir, exist_ok=True)

    return args


def function_accepts_argument(func, arg_name):
    signature = inspect.signature(func)
    return arg_name in signature.parameters


def get_attribution_method(method: str):
    attributions_map = {
        "ig": IntegratedGradientExplainer,
        "gradient": GradientExplainer,
        "lime": LIMEExplainer,
        "shap": SHAPExplainer,
    }
    if method not in attributions_map:
        raise ValueError(f"Attribution method {method} not supported, \
                         available methods: {attributions_map.keys()}")

    return attributions_map[method]


def ferret_interpret_model(model, tokenizer, dataset, 
                           label_key: str='label', attr_method: str='gradient') -> List:

    # ig = IntegratedGradientExplainer(model, tokenizer, multiply_by_inputs=True)
    # g = GradientExplainer(model, tokenizer, multiply_by_inputs=True)
    # l = LIMEExplainer(model, tokenizer)
    # s = SHAPExplainer(model, tokenizer)

    attr_method_cls = get_attribution_method(attr_method)
    if function_accepts_argument(attr_method_cls.__init__, "multiply_by_inputs"):
        explainer = attr_method_cls(model, tokenizer, multiply_by_inputs=True)
    else:
        explainer = attr_method_cls(model, tokenizer)

    bench = Benchmark(model, tokenizer,explainers=[explainer])

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


def main():
    args = attr_parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    logger.info(f"Using device: {DEVICE}")
    logger.info('args: %s', args)

    if args.seed is not None:
        set_seed(args.seed)

    print("start loading the dataset", "path:", args.dataset_path)
    all_dataset = get_csv_dataset(data_name=args.dataset_name, 
                                  model_name=args.model_name, 
                                  path=args.dataset_path, split="all",
                                  label_key=args.label_key, 
                                  use_fast=args.fast_tokenizer,)
    print("finished loading the dataset")
    if args.num_samples is not None:
        # randomly select num_samples from the dataset
        all_dataset = all_dataset.shuffle(seed=args.seed).select(range(args.num_samples))
        print(f"Using only {args.num_samples} samples from the dataset for attribution")

    model_name = args.model_name

    label_key = args.label_key

    num_labels = len(all_dataset.features[label_key].names)
    if args.num_labels is not None:
        num_labels = args.num_labels

    print("start constructing the model", "model_name:", model_name)
    model = construct_model(model_name=model_name, num_labels=num_labels).to(DEVICE)
    print("finished constructing the model")

    if args.load_lora:
        model = load_lora_model(model=model, 
                                target_modules=args.target_modules, 
                                task_type=TaskType.SEQ_CLS)


    attr_method = get_attribution_method(args.attr_method)


    print(model)

    # get model layer 
    layers = args.attr_target_layer.split(",")
    print("Layers:", layers, "embedding_name:", args.embeddings)
    target_layers = []
    for layer in layers:
        # get the layer from the model
        target_layer = get_nested_attr(model, layer)
        # set the target layer required gradient
        target_layer.requires_grad = True
        print("Target layer:", target_layer)
        if target_layer is None:
            raise ValueError(f"Layer {layer} not found in the model")
        target_layers.append(target_layer)

    print(attr_method, target_layers)

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              use_fast=args.fast_tokenizer,)

    
    print(args.checkpoint_dir)

    # if args.checkpoint_dir is not None:
    # check if the model checkpoint exists
    expected_checkpoint = os.path.join(args.checkpoint_dir, "model.pth")
    if args.checkpoint_dir is not None and os.path.exists(expected_checkpoint):
        # load the model from the checkpoint
        state_dict = torch.load(expected_checkpoint, map_location=DEVICE)
        # debug state_dict
        model.load_state_dict(state_dict=state_dict)
    else:
        logger.info("No checkpoint is provided, zero-shot evaluation")


    eval_outputs = ferret_interpret_model(model, tokenizer, all_dataset, 
                                          batch_size=args.eval_batch_size, debug=True,
                                          label_key=label_key)
    
    # get path basename from the dataset path
    dataset_basename = os.path.basename(args.dataset_path)

    # save pickle file
    filepath = os.path.join(args.output_attr_dir, f"ferret-output-{dataset_basename}.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(eval_outputs, f)


if __name__ == "__main__":
    main()



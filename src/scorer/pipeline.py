import json
import random
from typing import Optional
from typing import List

from datasets import ClassLabel
from datasets import load_dataset

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModel, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import AutoTokenizer


# from torch_geometric.data import Data as PyGData
# from torch_geometric.data import Dataset as PyGDataset
from torch_geometric.utils import k_hop_subgraph

# Copied from https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py.
GLUE_TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "custom_nli": ("premise", "hypothesis"),
    "custom_cls": ("hypothesis", None),
    "classification": ("premise", None),
}

MNLI_LABELS_3 = {"entailment", "neutral", "contradiction"}
BINARY_NLI_LABELS = {"entailment", "not_entailment"}


def construct_model(data_name: str = "sst2", model_name: str = "bert-base-cased", num_labels=2, 
                    tokenizer=None, device='cpu') -> nn.Module:
    # reserve max memory here
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        finetuning_task=data_name,
        trust_remote_code=True,
    )

    if tokenizer is not None and 'Qwen2' in model_name:
        config.eos_token_id = tokenizer.eos_token_id
        config.save_pretrained("./fixed_model_config")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        from_tf=False,
        config=config,
        # ignore_mismatched_sizes=False,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
        device_map=device,
    )

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        # Set the pad token to the EOS token (or any appropriate token for your model)
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id

    return model

def construct_generative_model(model_name: str = "facebook/bart-large", data_name: str=None) -> nn.Module:
    config = AutoConfig.from_pretrained(
        model_name,
        finetuning_task=data_name,
        trust_remote_code=True,
    )
    return AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        from_tf=False,
        config=config,
        trust_remote_code=True,
    )
    

def construct_model_auto(data_name: str = "sst2", model_name: str = "bert-base-cased") -> nn.Module:
    config = AutoConfig.from_pretrained(
        model_name,
        finetuning_task=data_name,
        trust_remote_code=True,
    )
    return AutoModel.from_pretrained(
        model_name,
        from_tf=False,
        config=config,
        # ignore_mismatched_sizes=False,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
    )

def construct_conditional_generation_model(model_name: str = "google-t5/t5-small") -> nn.Module:
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    return AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        from_tf=False,
        config=config,
        trust_remote_code=True,
    )

def construct_causal_lm(model_name: str, data_name: str = "esnli") -> nn.Module:
    config = AutoConfig.from_pretrained(
        model_name,
        finetuning_task=data_name,
        trust_remote_code=True,
    )
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        from_tf=False,
        config=config,
        trust_remote_code=True,
    )


def get_glue_dataset(
    data_name: str,
    split: str,
    indices: List[int] = None,
    model_name: str = "bert-base-cased"
) -> Dataset:
    assert split in ["train", "eval_train", "valid"]

    raw_datasets = load_dataset(
        path="glue",
        name=data_name,
    )
    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)
    # assert num_labels == 2

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[data_name]
    padding = "max_length"
    max_seq_length = 128

    def preprocess_function(examples):
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=max_seq_length, truncation=True)
        if "label" in examples:
            result["labels"] = examples["label"]
        return result

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
    )

    if split in ["train", "eval_train"]:
        train_dataset = raw_datasets["train"]
        ds = train_dataset
        if data_name == "rte":
            ds = ds.select(range(2432))
    else:
        eval_dataset = raw_datasets["validation"]
        ds = eval_dataset

    if indices is not None:
        ds = ds.select(indices)

    return ds


def get_hf_dataset(
    data_name: str,
    subset: str,
    split: str,
    indices: List[int] = None,
    model_name: str = "bert-base-cased",
    label_key: str = "label",
    max_seq_length: int = 128,
    padding: str = "max_length",
) -> Dataset:
    raw_datasets = load_dataset(
        data_name, subset
    )

    from datasets import ClassLabel

    # Get unique labels from the dataset
    unique_labels = sorted(set(raw_datasets["train"][label_key]))

    # Create a ClassLabel feature
    class_label = ClassLabel(names=unique_labels)

    # Cast the label_key column to ClassLabel
    raw_datasets = raw_datasets.cast_column(label_key, class_label)
    label_list = raw_datasets["train"].features[label_key].names
    print("Label names:", label_list)

    label_list = raw_datasets["train"].features[label_key].names
    num_labels = len(label_list)
    # assert num_labels == 2

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add pad token
        tokenizer.pad_token = tokenizer.eos_token

    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[data_name]
    padding = "max_length"

    def preprocess_function(examples):
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=max_seq_length, truncation=True)
        if label_key in examples:
            result["labels"] = examples[label_key]
        # add other columns
        for col in examples:
            if col not in result:
                result[col] = examples[col]
        return result

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
    )

    if split in ["train", "eval_train"]:
        train_dataset = raw_datasets["train"]
        ds = train_dataset
        if data_name == "rte":
            ds = ds.select(range(2432))
    else:
        eval_dataset = raw_datasets["validation"]
        ds = eval_dataset

    if indices is not None:
        ds = ds.select(indices)

    return ds


def align_labels_to_model(label_list, model_config) -> List[str]:
    """
    Align dataset labels to model’s existing id2label if possible.
    Returns reordered list or original if no match.
    """
    if not model_config.id2label:
        return label_list
    model_labels = [model_config.id2label[i].lower() for i in range(len(model_config.id2label))]
    lower_dataset = [l.lower() for l in label_list]
    # Exact set match ignoring case
    if set(lower_dataset) == set(model_labels):
        # Reorder to model label order
        return [l for l in model_labels]
    return label_list


def get_csv_dataset(
    data_name: str,
    path: str,
    split: str,
    indices: List[int] = None,
    model_name: str = "bert-base-cased",
    label_key: str = "label",
    val_path: str = None,
    test_path: str = None,
    max_seq_length: int = 256,
    padding: str = "max_length",
    tokenizer=None,
    filter_function=None,
    use_fast: bool = True,
    augment_ratio: float = 1.0,
    # sep: str = ",",
) -> Dataset:
    data_files = {"train": path}
    if val_path is not None:
        data_files["val"] = val_path
    if test_path is not None:
        data_files["test"] = test_path
    raw_datasets = load_dataset(
        "csv",
        name=data_name,
        data_files=data_files,
        # sep=sep
    )

    # if model name contains 'mnli', set num_labels to 3

    if augment_ratio < 1.0 and 'train' in raw_datasets:
        # reduce the training dataset size
        train_size = int(len(raw_datasets['train']) * augment_ratio)
        raw_datasets['train'] = raw_datasets['train'].select(range(train_size))
        print(f"Reduced the dataset size to {train_size}")

    # Get unique labels from the dataset
    unique_labels = sorted(set(raw_datasets["train"][label_key]))

    if "mnli" in model_name:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        ordered_labels = align_labels_to_model(unique_labels, config)
        class_label = ClassLabel(names=ordered_labels)
    else:
        # Create a ClassLabel feature
        class_label = ClassLabel(names=unique_labels, num_classes=len(unique_labels))

    # Cast the label_key column to ClassLabel
    raw_datasets = raw_datasets.cast_column(label_key, class_label)
    label_list = raw_datasets["train"].features[label_key].names
    print("Label names:", label_list)

    label_list = raw_datasets["train"].features[label_key].names
    num_labels = len(label_list)
    # assert num_labels == 2

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast, 
                                                  trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add pad token
        tokenizer.pad_token = tokenizer.eos_token

    # check if model name contain Qwen2
    if "Qwen2" in model_name:
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.pad_token = tokenizer.eos_token  # If pad_token is missing
        tokenizer.save_pretrained("./fixed_tokenizer")


    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[data_name]
    # padding = "max_length"

    def preprocess_function(examples):
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        # print(f"Processing examples: {texts}")
        result = tokenizer(*texts, padding=padding, 
                           max_length=max_seq_length, truncation=True)
        # print(f"Result keys: {result.keys()}")
        if label_key in examples:
            result["labels"] = examples[label_key]
            # mapped_label = class_label.str2int(examples[label_key])
            # result["labels"] = mapped_label
        # add other columns
        for col in examples:
            # print(col)
            if col not in result and col != 'label' :
                result[col] = examples[col]
        return result

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        # batch_size=batch_size,
        load_from_cache_file=True,
    )

    # if label_key is not 'label' remove the label column
    if label_key != 'label' and 'label' in raw_datasets["train"].features:
        raw_datasets = raw_datasets.remove_columns(['label'])

    # shuffle the dataset
    raw_datasets = raw_datasets.shuffle(seed=42)

    # Split the dataset into train, eval_train, and test
    train_test_split = raw_datasets['train'].train_test_split(test_size=0.2, seed=42)  # 80% train, 20% test
    train_eval_split = train_test_split['train'].train_test_split(test_size=0.1, seed=42)  # 10% of train for eval_train

    eval_train_dataset = None
    test_dataset = None
    # Combine splits
    train_dataset = train_eval_split['train']  # Training set
    if val_path is None:
        eval_train_dataset = train_eval_split['test']  # Evaluation training set
    else:
        eval_train_dataset = raw_datasets['val']  # Evaluation training set
    
    if test_path is None:
        test_dataset = train_test_split['test']
    else:
        test_dataset = raw_datasets['test']

    # Print the sizes of each split
    print(f"Train size: {len(train_dataset)}")
    print(f"Eval train size: {len(eval_train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    # print(raw_datasets["train"].features)
    # print raw_datasets 

    # print the dataset label mapping
    print(raw_datasets["train"].features[label_key].int2str)
    # print(raw_datasets["train"].features['label'])

    if split in ["train", "eval_train"]:
        train_dataset = train_dataset
        ds = train_dataset
        if data_name == "rte":
            ds = ds.select(range(2432))
    elif split == "validation":
        eval_dataset = eval_train_dataset
        ds = eval_dataset
    elif split == "test":
        ds = test_dataset
    elif split == "all":
        ds = raw_datasets["train"]
    else:
        eval_dataset = test_dataset
        ds = eval_dataset

    if indices is not None:
        ds = ds.select(indices)

    if filter_function is not None:
        # apply filter function
        ds = ds.filter(filter_function)

    return ds

if __name__ == "__main__":
    from kronfluence import Analyzer

    model = construct_model()
    print(Analyzer.get_module_summary(model))

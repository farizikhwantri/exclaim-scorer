import os
import csv
import argparse
import logging
from tqdm import tqdm
import pickle
import random

import inspect

import pandas as pd

import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from sklearn.metrics import classification_report

from inseq import load_model

from src.scorer.explainer_llm import LLMAttribution
from src.scorer.faithfulness_lm import AOPC_Comprehensiveness_LLM_Evaluation
from src.scorer.faithfulness_lm import AOPC_Sufficiency_LLM_Evaluation

from src.scorer.utils_prompt import build_prompt
from src.scorer.utils_prompt import build_prompt_curried
from src.scorer.utils_prompt import zero_shot_prompt
from src.scorer.utils_prompt import multi_hop_prompt
from src.scorer.utils_prompt import multi_hop_prompt_curried
from src.scorer.utils_prompt import task_prefix

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# check device
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
print(f"Using device: {DEVICE}")
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a causal LLM for NLI entailment using in-context learning / zero-shot classification")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the CSV dataset file")
    parser.add_argument("--train_path", type=str, default=None, help="Path to the training dataset file (optional)")
    parser.add_argument("--zero_shot", action="store_true", help="Use zero-shot classification")
    parser.set_defaults(zero_shot=False)
    parser.add_argument("--multi_hop", action="store_true", help="Use multi-hop classification")
    parser.set_defaults(multi_hop=False)
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the predictions")
    parser.add_argument("--model_type", type=str, default="causal", choices=["causal", "seq2seq"], help="Model type: causal or seq2seq")
    parser.add_argument("--model_name", type=str, default="gpt2-xl", help="Causal LLM model name (>1B parameters)")
    parser.add_argument("--attr_method", type=str, default="integrated_gradients", help="Attribution method to use")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--max_length", type=int, default=64, help="Max generation length")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of examples per batch (processing sequentially)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--task_prefix", type=str, default=None, help="Task description prefix")
    parser.set_defaults(debug=False)
    args = parser.parse_args()
    return args

def load_csv_dataset(csv_path):
    dataset = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Expecting keys: premise, hypothesis, label (optional)
            instance ={
                "premise": row["premise"],
                "hypothesis": row["hypothesis"],
                "label": row.get("label", "").lower()  # can be 'entailment' or other
            }
            # copy the rest of the row
            for key, value in row.items():
                if key not in instance:
                    instance[key] = value
            dataset.append(instance)
    return dataset

def classify_entailment(generated_text):
    # Postprocess: if generated text (lowercase) contains 'entail', mark as entailment.
    # Otherwise choose not entailment.
    text = generated_text.strip().lower()
    if "not" in text or "not entail" in text:
        return "not_entailment"
    elif "ent" in text or "entail" in text:
        return "entailment"
    return "not_entailment"

def evaluate_model(model, tokenizer, dataset, max_length, bulild_prompt_func=build_prompt, 
                   model_type="causal", attr_method="integrated_gradients", debug=False,
                   task_prefix=None):
    model.eval()

    # Create LLMAttribution explainer.
    # debug print model device
    print(f"Model device: {model.device}")
    explainer = LLMAttribution(model, tokenizer, attribution_method=attr_method)
    # debug print explainer device
    # print(f"Explainer device: {explainer.device}")

    aopc_comp_eval = AOPC_Comprehensiveness_LLM_Evaluation(model, tokenizer)
    aopc_suff_eval = AOPC_Sufficiency_LLM_Evaluation(model, tokenizer)

    all_predictions = []
    all_labels = []

    eval_results = []
    expl_results = []

    # Define removal arguments (for demonstration, use threshold based approach and masking)
    removal_args = {
        # "based_on": "th",       # using a threshold method
        # "thresholds": [0.5],    # a list of threshold values
        "remove_tokens": False  # set False to use masking (mask token) instead of deletion
    }
    evaluation_args = {"removal_args": removal_args, "remove_first_last": False, "only_pos": True}

    for example in tqdm(dataset, desc="Evaluating"):
        # prompt = bulild_prompt_func(example["premise"], example["hypothesis"])
        # prompt = bulild_prompt_func(example)
        # check if the build_prompt_func accepts task_prompt or two arguments
        sig = inspect.signature(bulild_prompt_func)
        if "task_prompt" in sig.parameters and task_prefix is not None:
            prompt = bulild_prompt_func(example, task_prompt=task_prefix)
        else:
            prompt = bulild_prompt_func(example)

        assert prompt is not None, "Prompt should not be None"
        if debug:
            print("------------------------")
            print(prompt)

        # inputs = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

        # with torch.no_grad():
        #     outputs = model.generate(
        #         inputs,
        #         max_length=inputs.shape[1] + max_length,
        #         do_sample=True,
        #         pad_token_id=tokenizer.eos_token_id,
        #     )

        inputs = tokenizer(prompt, return_tensors="pt")
        # Detect the first device the model is on (works for parallelized models)
        first_param_device = next(model.parameters()).device
        inputs = {k: v.to(first_param_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + max_length,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if debug:
            print("------------------------")
            print(generated)

        # extract generated answer after the prompt
        # answer = generated[len(prompt):].strip().split()[0]  # first token word

        if model_type == "causal":
            answer = generated[len(prompt):].strip().split()
            if len(answer) > 0:
                answer = answer[0]
            else:
                # if no answer is generated, set to not entailment
                answer = "not entailment"
        elif model_type == "seq2seq":  
            answer = generated
        pred = classify_entailment(answer)
        all_predictions.append(pred)
        if example["label"]:
            all_labels.append(example["label"])
        # evaluate the faithfulness of the generated answer

        model.train()  # Set model back to training mode

        if generated is None or len(generated) == 0:
            generated = "not entailment"

        explanations = explainer.compute_feature_importance(
            prompt, target=1, generated_texts=None, n_steps=5, step_scores=["logit"],
            include_eos_baseline=True, 
            output_step_attributions=True,
            max_new_tokens=max_length,
        )

        # take generated text from the last explanation step
        generated = explanations[-1].text
        if debug:
            print("------------------------")
            print(generated)

        # extract generated answer after the prompt
        answer = generated[len(prompt):].strip().split()[0]  # first token word

        if model_type == "causal":
            answer = generated[len(prompt):].strip().split()
            if len(answer) > 0:
                answer = answer[0]
            else:
                # if no answer is generated, set to not entailment
                answer = "not entailment"
        elif model_type == "seq2seq":  
            answer = generated
        pred = classify_entailment(answer)
        all_predictions.append(pred)
        if example["label"]:
            all_labels.append(example["label"])
        # create dummy explanation with zero scores same length as generated text
        # explanations = [Explanation(text=generated, 
        #                             scores=np.zeros(len(generated.split())), 
        #                             tokens=generated.split(), 
        #                             explainer="dummy", target="not entailment")]

        expl_results.append(explanations)

        # print("Explanations:", explanations, len(explanations))
        # print(explanations[-1].text)
        
        # generated = [explanation.target for explanation in explanations]
        # # convert the generated text to id and then decode it
        # generated_id = [tokenizer.convert_tokens_to_ids(explanation.target) for explanation in explanations]
        # # print("Generated IDs:", generated_id)
        # generated = tokenizer.decode(generated_id, skip_special_tokens=True)
        # print(generated)

        # generated = explanations[-1].text

        average_comprehensiveness = 0
        average_sufficiency = 0
        for _, expl in enumerate(explanations):
            # print(f"Explanation {i}:")
            # print("Text:", expl.text)
            # print("Tokens:", [t for t in expl.tokens])
            # print("Scores shape:", np.array(expl.scores))
            # print("Explainer:", expl.explainer)
            # print("Target token:", expl.target)
            # print("\n")
            compr = aopc_comp_eval.compute_evaluation(expl, expl.target, token_position=None, **evaluation_args)
            suff = aopc_suff_eval.compute_evaluation(expl, expl.target, token_position=None, **evaluation_args)
            # print(f"Comprehensiveness: {compr}, Sufficiency: {suff}")
            average_comprehensiveness += compr.score
            average_sufficiency += suff.score

        average_comprehensiveness /= len(explanations)
        average_sufficiency /= len(explanations)

        print(f"Average Comprehensiveness: {average_comprehensiveness}, Average Sufficiency: {average_sufficiency}")

        res = {
            "premise": example["premise"],
            "hypothesis": example["hypothesis"],
            "predicted_label": pred,
            "true_label": example.get("label", ""),
            "average_comprehensiveness": average_comprehensiveness,
            "average_sufficiency": average_sufficiency
        }

        # copy the example metadata
        for key, value in example.items():
            if key not in res:
                res[key] = value

        eval_results.append(res)

    return eval_results, expl_results, all_predictions, all_labels

def load_large_model(model_name_or_path, load_in_8bit=False, load_in_4bit=False):
    """
    Load a large language model across multiple GPUs
    """
    print(f"Loading model: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Set specific parameters for large models
    kwargs = {
        "device_map": "auto",       # Automatically distribute across GPUs
        # "torch_dtype": torch.float16,  # Use half precision
        "trust_remote_code": True,
    }
    
    # Add quantization options if requested
    if load_in_8bit:
        print("Using 8-bit quantization")
        kwargs["load_in_8bit"] = True
    elif load_in_4bit:
        print("Using 4-bit quantization")
        kwargs["load_in_4bit"] = True
        kwargs["bnb_4bit_quant_type"] = "nf4"
        kwargs["bnb_4bit_compute_dtype"] = torch.float16
    
    # Load the model with the specified parameters
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **kwargs
    )
    
    # Print memory usage per GPU
    print("\nMemory usage after loading:")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
    
    return model, tokenizer

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.info("Arguments: %s", args)
    
    if args.seed is not None:
        set_seed(args.seed)

    # if task prefix is not None, check to task prefix dict
    if args.task_prefix is not None:
        args.task_prefix = task_prefix.get(args.task_prefix, "gdpr")
    
    logger.info("Loading dataset from: %s", args.dataset_path)
    dataset = load_csv_dataset(args.dataset_path)
    logger.info("Dataset size: %d", len(dataset))
    
    # logger.info("Loading model: %s", args.model_name)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    model = None
    # if args.model_type == "causal":
    #     model = AutoModelForCausalLM.from_pretrained(args.model_name).to(DEVICE)
    # elif args.model_type == "seq2seq":
    #     model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(DEVICE)
    # model = load_model(
    #         model=args.model_name,
    #         attribution_method=args.attr_method,
    #         model_kwargs={"device_map": DEVICE, "torch_dtype": torch.float16},
    #         tokenizer_kwargs={"use_fast": False},
    #     )
    model, tokenizer = load_large_model(
        model_name_or_path=args.model_name, 
        load_in_8bit=False,  # Set to True if you want to load in 8-bit
        load_in_4bit=False   # Set to True if you want to load in 4-bit
    )

    # Ensure the tokenizer has an eos_token
    # Ensure that the tokenizer has an unk_token; if not, add one.
    if tokenizer.unk_token is None:
        tokenizer.unk_token = tokenizer.eos_token

    # If the mask token is missing, define it (or use the unk token as fallback)
    if tokenizer.mask_token is None:
        tokenizer.mask_token = tokenizer.pad_token

    # Similarly, ensure the pad token is defined:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    build_prompt_func = build_prompt

    if args.train_path:
        train_dataset = load_csv_dataset(args.train_path)
        # choose randomly two examples from the training dataset of positive and negative labels
        # random sample two examples
        pos_example, neg_example = None, None
        while True:
            pos_example = train_dataset[torch.randint(0, len(train_dataset), (1,)).item()]
            if pos_example["label"] == "entailment":    
                pos_example = (pos_example["premise"], pos_example["hypothesis"])
                break
        while True:
            neg_example = train_dataset[torch.randint(0, len(train_dataset), (1,)).item()]
            if neg_example["label"] != "entailment":   
                neg_example = (neg_example["premise"], neg_example["hypothesis"]) 
                break
        build_prompt_func = build_prompt_curried(pos_example, neg_example)
    if args.zero_shot:
        build_prompt_func = zero_shot_prompt
    
    if args.multi_hop:
        build_prompt_func = multi_hop_prompt
        if args.train_path:
            train_dataset = load_csv_dataset(args.train_path)
            # choose randomly two examples from the training dataset of positive and negative labels
            # random sample two examples
            pos_example, neg_example = None, None
            while True:
                pos_example = train_dataset[torch.randint(0, len(train_dataset), (1,)).item()]
                if pos_example["label"] == "entailment":    
                    pos_example = (pos_example["premise"], pos_example["hypothesis"])
                    break
            while True:
                neg_example = train_dataset[torch.randint(0, len(train_dataset), (1,)).item()]
                if neg_example["label"] != "entailment":   
                    neg_example = (neg_example["premise"], neg_example["hypothesis"]) 
                    break
            build_prompt_func = multi_hop_prompt_curried(pos_example, neg_example)

    not_entailment_length = tokenizer("not entailment", return_tensors="pt").input_ids.shape[1]
    entailment_length = tokenizer("entailment", return_tensors="pt").input_ids.shape[1]

    max_length = max(not_entailment_length, entailment_length)

    max_length = args.max_length
    logger.info("Max generation length: %d", max_length)

    # if debug use only 10 examples
    if args.debug:
        # dataset = dataset[:10]
        # get 10% number of examples from the dataset
        num_samples = int(len(dataset) * 0.1) if args.num_samples is None else args.num_samples
        if args.multi_hop:
            # pick examples with multi-hop > 1
            # assert int(dataset[0].get("hop", 1)) >= 1, "dataset should have hop > 1"
            # dataset = [ex for ex in dataset if int(ex.get("hop", 1)) > 1]
            # logger.info("Debug mode: using only examples with multi-hop > 1")
            # print(len(dataset))
            random_indices = torch.randint(0, len(dataset), (num_samples,))
            dataset = [dataset[i] for i in random_indices]
        else:
            # pick randomly 10 examples
            random_indices = torch.randint(0, len(dataset), (num_samples,))
            dataset = [dataset[i] for i in random_indices]
            logger.info("Debug mode: using only examples")

    eval_res, expl_res, preds, labels = evaluate_model(model, tokenizer, dataset, max_length, build_prompt_func, 
                                                       attr_method=args.attr_method,
                                                       model_type=args.model_type, debug=args.debug,
                                                       task_prefix=args.task_prefix)
    # If labels exist, print a simple report
    if labels:
        report = classification_report(labels, preds, zero_division=0)
        print("Classification Report:\n", report)
    else:
        logger.info("Zero-shot predictions (first 10): %s", preds[:10])

    # Save predictions to output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, "predictions.csv")
        df = pd.DataFrame(eval_res)
        df.to_csv(output_file, index=False, encoding="utf-8")
        logger.info("eval saved to: %s", output_file)

        # saved the explanations to a file
        expl_output_file = os.path.join(args.output_dir, "explanations.pickle")
        with open(expl_output_file, "wb") as f:
            pickle.dump(expl_res, f)
        logger.info("explanations saved to: %s", expl_output_file)
    
if __name__ == "__main__":
    main()

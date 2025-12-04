import copy
import numpy as np
from scipy.stats import kendalltau
from statistics import mean

import torch

from ferret.model_utils import ModelHelper
from ferret.evaluators.evaluation import Evaluation
from ferret.evaluators import BaseEvaluator
from ferret.explainers.explanation import Explanation
from ferret.evaluators.utils_from_soft_to_discrete import (
    _check_and_define_get_id_discrete_rationale_function,
    parse_evaluator_args,
)

def _compute_aopc(scores):
    return mean(scores)

# LLM helper to wrap model and tokenizer (providing _forward and _tokenize methods)
class LLMHelper():
    def __init__(self, model, tokenizer, device='cpu'):
        """
        Initialize the LLM helper with a model and tokenizer.
        
        Parameters:
            model: A causal LM model.
            tokenizer: The corresponding tokenizer.
        """
        # super().__init__(model, tokenizer)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _forward(self, text, output_hidden_states=False, **tok_kwargs):
        # If text is a list, forward all examples
        # print("text:", text)
        self.model.eval()  # Set the model to evaluation mode
        # empty cuda grad to optimize memory usage
        torch.cuda.empty_cache()
        if isinstance(text, list):
            logits_list = []
            for t in text:
                # torch no grad
                inputs = self._tokenize(t, return_tensors="pt", padding=True, **tok_kwargs)
                # move inputs to the same device as the model
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                outputs = self.model(**inputs, output_hidden_states=output_hidden_states)
                logits_list.append(outputs.logits)
            return None, logits_list
        else:
            inputs = self._tokenize(text, return_tensors="pt", padding=True, **tok_kwargs)
            # move inputs to the same device as the model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model(**inputs, output_hidden_states=output_hidden_states)
            return None, outputs.logits

    def _tokenize(self, text: str, return_tensors="pt", padding=True, tok_kwargs=None):
        """
        Base tokenization strategy for a single text.

        Note that we truncate to the maximum length supported by the model.

        :param text str: the string to tokenize
        """
        tok_kwargs = tok_kwargs or {}
        tok_kwargs["return_tensors"] = return_tensors
        return self.tokenizer(text, padding=padding, truncation=True, **tok_kwargs)


class AOPC_Comprehensiveness_LLM_Evaluation(BaseEvaluator):
    NAME = "aopc_comprehensiveness_llm"
    SHORT_NAME = "aopc_compr_llm"
    BEST_SORTING_ASCENDING = False
    TYPE_METRIC = "faithfulness"

    def __init__(self, model, tokenizer):
        """
        Initialize the AOPC comprehensiveness evaluator for LLMs.
        
        Parameters:
            model: A causal LM model.
            tokenizer: The corresponding tokenizer.
        """
        super().__init__(model, tokenizer)
        self.helper = LLMHelper(model, tokenizer)
        # Set the tokenizer's mask token if not already set

    def compute_evaluation(self, explanation: Explanation, target_token: str, token_position: int = None, **evaluation_args):
        """
        Compute AOPC comprehensiveness for LLMs.
          - The baseline is computed by selecting the probability of target_token
            at a specific output position (default: the last token).
          - For each rationale (set of token indices determined via a threshold),
            we remove those tokens from the input and recompute the probability.
          - The average drop in probability (baseline minus new probability) is the AOPC score.
        """
        remove_first_last, only_pos, removal_args, _ = parse_evaluator_args(evaluation_args)
        text = explanation.text
        score_explanation = explanation.scores

        # Forward pass: For LLMs, logits shape is [batch, seq_len, vocab_size]
        _, logits = self.helper._forward(text, output_hidden_states=False)
        pos = None
        if token_position is None:
            # print("Logits shape:", logits.shape)
            pos = logits.shape[1] - 1  # use last token by default
        target_id = self.tokenizer.convert_tokens_to_ids(target_token)
        baseline = logits.softmax(-1)[0, pos, target_id].item()
        # print("Baseline probability for token '{}' at position {}: {:.4f}".format(
        #     target_token, token_position, baseline))

        if self.helper.tokenizer.mask_token is None:
            self.helper.tokenizer.mask_token = self.helper.tokenizer.unk_token

        # Tokenize to get input ids
        item = self.helper._tokenize(text)
        input_len = item["attention_mask"].sum().item()
        input_ids = item["input_ids"][0][:input_len].tolist()
        if remove_first_last:
            input_ids = input_ids[1:-1]
            if self.tokenizer.cls_token == explanation.tokens[0]:
                score_explanation = score_explanation[1:-1]

        # Determine the minimum length between token ids and saliency scores
        # assert len(input_ids) == len(score_explanation), \
        #     f"Input IDs and score explanation lengths do not match. {len(input_ids)} vs {len(score_explanation)}, {len(explanation.tokens)}"
        if len(input_ids) != len(score_explanation):
            assert len(score_explanation) == len(explanation.tokens), \
                f" {len(score_explanation)} vs {len(explanation.tokens)}"
            
            input_ids = self.tokenizer.convert_tokens_to_ids(explanation.tokens)
            # print(type(input_ids), type(score_explanation))
            input_len = len(input_ids)

        if len(input_ids) < len(score_explanation):
            min_len = min(len(input_ids), len(score_explanation))
            # Truncate both so that they align
            item["attention_mask"] = item["attention_mask"][:, :min_len]
            item["input_ids"] = item["input_ids"][:, :min_len]
            input_ids = input_ids[:min_len]
            score_explanation = score_explanation[:min_len]

        # Ensure score_explanation is a 1D tensor (remove the batch dimension if present)
        # print("Score explanation shape:", score_explanation.shape)
        if score_explanation.ndim > 1:
            score_explanation = score_explanation.squeeze(0)

        discrete_expl_ths = []
        id_tops = []
        get_discrete_rationale_function = _check_and_define_get_id_discrete_rationale_function(
            removal_args["based_on"]
        )

        thresholds = removal_args["thresholds"]
        last_id_top = None
        for v in thresholds:
            # Get the rationale indices based on a threshold (or other approach)
            # print(score_explanation.shape, v, only_pos)
            id_top = get_discrete_rationale_function(score_explanation, v, only_pos)
            # print('id_top:', id_top)
            # print("Rationale indices for threshold {}: {}".format(v, id_top))
            if id_top is not None and last_id_top is not None and set(id_top) == last_id_top:
                id_top = None
            id_tops.append(id_top)
            if id_top is None:
                continue
            last_id_top = set(id_top)
            # Comprehensiveness: Remove the tokens in the rationale
            sample = np.array(copy.copy(input_ids))
            if removal_args["remove_tokens"]:
                discrete_expl_th_token_ids = np.delete(sample, id_top)
            else:
                # print(len(sample), len(score_explanation), max(id_top))
                sample[id_top] = self.tokenizer.mask_token_id
                discrete_expl_th_token_ids = sample
            # print("len(discrete_expl_th_token_ids):", len(discrete_expl_th_token_ids))
            discrete_expl_th = self.tokenizer.decode(discrete_expl_th_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            discrete_expl_ths.append(discrete_expl_th)

        # If no rationale was detected, return a score of 0 (best)
        if len(discrete_expl_ths) == 0:
            return Evaluation(self.SHORT_NAME, 0)

        # Forward pass on each modified text
        # print("Discrete explanations:", len(discrete_expl_ths))
        # print('type helper:', type(self.helper))
        _, logits_removed = self.helper._forward(discrete_expl_ths, output_hidden_states=False)
        probs_removed = []
        for logit in logits_removed:
            logit_soft = logit.softmax(-1)
            # print("Logit shape:", logit_soft.shape, pos, target_id)
            if logit_soft.ndim == 3:
                pos = logit_soft.shape[1] - 1  # pick the last token
                if token_position:
                    pos = token_position
                probs_removed.append(logit_soft[0, pos, target_id].item())
            else:
                pos = logit_soft.shape[0] - 1  # pick the last token
                if token_position:
                    pos = token_position
                probs_removed.append(logit_soft[pos, target_id].item())
        probs_removed = np.array(probs_removed)
        removal_importance = baseline - probs_removed
        # to cpu
        if isinstance(removal_importance, torch.Tensor):
            removal_importance = removal_importance.cpu().numpy()
        aopc_comprehensiveness = _compute_aopc(removal_importance)
        return Evaluation(self.SHORT_NAME, aopc_comprehensiveness)


class AOPC_Sufficiency_LLM_Evaluation(BaseEvaluator):
    NAME = "aopc_sufficiency_llm"
    SHORT_NAME = "aopc_suff_llm"
    BEST_SORTING_ASCENDING = True
    TYPE_METRIC = "faithfulness"

    def __init__(self, model, tokenizer):
        """
        Initialize the AOPC comprehensiveness evaluator for LLMs.
        
        Parameters:
            model: A causal LM model.
            tokenizer: The corresponding tokenizer.
        """
        super().__init__(model, tokenizer)
        self.helper = LLMHelper(model, tokenizer)
        # Set the tokenizer's mask token if not already set

    def compute_evaluation(self, explanation: Explanation, target_token: str, token_position: int = None, **evaluation_args):
        """
        Compute AOPC sufficiency for LLMs.
          - The baseline is computed similarly by selecting the target token probability.
          - Here, we keep (rather than remove) only the tokens in the rationale, then
            compute the probability.
          - The average difference (baseline minus probability with only the rationale) is the AOPC sufficiency score.
        """
        remove_first_last, only_pos, removal_args, _ = parse_evaluator_args(evaluation_args)
        text = explanation.text
        score_explanation = explanation.scores

        _, logits = self.helper._forward(text, output_hidden_states=False)
        pos = None
        if token_position is None:
            pos = logits.shape[1] - 1
        target_id = self.tokenizer.convert_tokens_to_ids(target_token)
        baseline = logits.softmax(-1)[0, pos, target_id].item()
        # print("Baseline probability for token '{}' at position {}: {:.4f}".format(
        #     target_token, token_position, baseline))
        
        if self.helper.tokenizer.mask_token is None:
            self.helper.tokenizer.mask_token = self.helper.tokenizer.unk_token

        item = self.helper._tokenize(text)
        input_len = item["attention_mask"].sum().item()
        input_ids = item["input_ids"][0][:input_len].tolist()
        if remove_first_last:
            input_ids = input_ids[1:-1]
            if self.tokenizer.cls_token == explanation.tokens[0]:
                score_explanation = score_explanation[1:-1]

        if score_explanation.ndim > 1:
            score_explanation = score_explanation.squeeze(0)

        # assert len(input_ids) == len(score_explanation), \
        #     f"Input IDs and score explanation lengths do not match. {len(input_ids)} vs {len(score_explanation)}"
        if len(input_ids) != len(score_explanation):
            assert len(score_explanation) == len(explanation.tokens), \
                f" {len(score_explanation)} vs {len(explanation.tokens)}"
            
            input_ids = self.tokenizer.convert_tokens_to_ids(explanation.tokens)
            # print(type(input_ids), type(score_explanation))
            input_len = len(input_ids)

        if len(input_ids) < len(score_explanation):
            min_len = min(len(input_ids), len(score_explanation))
            # Truncate both so that they align
            item["attention_mask"] = item["attention_mask"][:, :min_len]
            item["input_ids"] = item["input_ids"][:, :min_len]
            input_ids = input_ids[:min_len]
            score_explanation = score_explanation[:min_len]

        discrete_expl_ths = []
        id_tops = []
        get_discrete_rationale_function = _check_and_define_get_id_discrete_rationale_function(
            removal_args["based_on"]
        )

        thresholds = removal_args["thresholds"]
        last_id_top = None
        for v in thresholds:
            id_top = get_discrete_rationale_function(score_explanation, v, only_pos)
            if id_top is not None and last_id_top is not None and set(id_top) == last_id_top:
                id_top = None
            id_tops.append(id_top)
            if id_top is None:
                continue
            last_id_top = set(id_top)
            # Sufficiency: Keep only the tokens in the rationale
            sample = np.array(copy.copy(input_ids))

            id_top = np.sort(id_top)
            non_top = np.setdiff1d(np.arange(len(sample)), id_top)

            if removal_args["remove_tokens"]:
                discrete_expl_th_token_ids = sample[id_top]
            else:
                # discrete_expl_th_token_ids = sample
                sample[non_top] = self.tokenizer.mask_token_id
                discrete_expl_th_token_ids = sample

            discrete_expl_th = self.tokenizer.decode(discrete_expl_th_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            discrete_expl_ths.append(discrete_expl_th)

        if len(discrete_expl_ths) == 0:
            return Evaluation(self.SHORT_NAME, 1)

        _, logits_removed = self.helper._forward(discrete_expl_ths, output_hidden_states=False)
        probs_removed = []
        for logit in logits_removed:
            logit_soft = logit.softmax(-1)
            if logit_soft.ndim == 3:
                pos = logit_soft.shape[1] - 1  # pick the last token
                if token_position:
                    pos = token_position
                probs_removed.append(logit_soft[0, pos, target_id].item())
            else:
                pos = logit_soft.shape[0] - 1  # pick the last token
                if token_position:
                    pos = token_position
                probs_removed.append(logit_soft[pos, target_id].item())
        probs_removed = np.array(probs_removed)
        removal_importance = baseline - probs_removed
        if isinstance(removal_importance, torch.Tensor):
            removal_importance = removal_importance.cpu().numpy()
        aopc_sufficiency = _compute_aopc(removal_importance)
        return Evaluation(self.SHORT_NAME, aopc_sufficiency)


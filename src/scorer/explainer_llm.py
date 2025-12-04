import copy
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

from inseq import load_model
from inseq.models import AttributionModel
from ferret.explainers import BaseExplainer
from ferret.explainers.explanation import Explanation

# from faithfulness_lm import AOPC_Comprehensiveness_LLM_Evaluation, AOPC_Sufficiency_LLM_Evaluation

class LLMAttribution(BaseExplainer):
    NAME = "LLMAttribution"

    def __init__(self, model, tokenizer, attribution_method="integrated_gradients"):
        """
        Initialize the LLMAttribution wrapper.
        
        Parameters:
            model: A causal LM model.
            tokenizer: The corresponding tokenizer.
            attribution_method: The inseq attribution method to use.
        """
        super().__init__(model, tokenizer)
        # Load the inseq attribution model using the provided attribution method.
        self.attribution_method = attribution_method
        if isinstance(model, AttributionModel):
            print("Using provided model.")
            self.inseq_model = model  # Assuming the model is already an inseq-compatible model
        else:
            # Load the model using inseq's load_model function.
            print(f"Loading inseq model with attribution method: {attribution_method}")
            self.inseq_model = load_model(model, 
                                          attribution_method=attribution_method,
                                          tokenizer=tokenizer,
                                          model_kwargs={"device_map": "auto",
                                                        #  "torch_dtype": torch.float16
                                          },
                                        #   model_kwargs={"device_map": 'cuda:0', "torch_dtype": torch.float16},
                                        #   tokenizer_kwargs={"use_fast": False},
                                        )
        self.inseq_model.tokenizer = tokenizer
        if self.inseq_model.tokenizer.pad_token is None:
            # Set the pad token to the tokenizer's eos token if pad token is not set.
            self.inseq_model.tokenizer.pad_token = self.inseq_model.tokenizer.eos_token
        if self.helper.tokenizer.pad_token is None:
            # Set the pad token to the tokenizer's eos token if pad token is not set.
            self.helper.tokenizer.pad_token = self.helper.tokenizer.eos_token
    
    def compute_feature_importance(self, text: str, target: int = 1, generated_texts=None, **explainer_args):
        """
        Compute token-level importance scores using inseq's integrated gradients and
        iterate over multiple attribution steps. Returns a list of Explanation objects,
        one per attribution step.
        
        Parameters:
          text (str): Input text to explain.
          target (int): (Ignored here; use explainer_args for target handling).
          **explainer_args: Additional arguments (e.g., n_steps, step_scores, 
                           include_eos_baseline, output_step_attributions).
        
        Returns:
          List[Explanation]: A list of Explanation objects for each attribution step.
        """
        # compute input token length
        inputs = self.tokenizer(text, return_tensors="pt")
        length = inputs["input_ids"].shape[1]
        default_max_new_tokens = length + 20

        # Get optional parameters with defaults
        n_steps = explainer_args.get("n_steps", 20)
        step_scores = explainer_args.get("step_scores", ["logit"])
        include_eos_baseline = explainer_args.get("include_eos_baseline", True)
        output_step_attributions = explainer_args.get("output_step_attributions", True)
        max_new_tokens = explainer_args.get("max_new_tokens", default_max_new_tokens)

        attribution_args = {}
        
        if self.attribution_method == "integrated_gradients":
            # For integrated gradients, we need to set the number of steps
            attribution_args["n_steps"] = n_steps

        max_length = max([default_max_new_tokens, max_new_tokens])
        
        # print helper tokenizer and inseq_model tokenizer
        # print(f"helper tokenizer: {self.tokenizer.pad_token_id}, {self.tokenizer.eos_token_id}")
        # print(f"inseq_model tokenizer: {self.inseq_model.tokenizer.pad_token_id}, {self.inseq_model.tokenizer.eos_token_id}")
        # Call the inseq attribution API
        attribution = self.inseq_model.attribute(
            text,
            # generated_texts=generated_texts,
            attribution_args=attribution_args,
            generation_args={
                "max_length": max_length,
                "pad_token_id": self.inseq_model.tokenizer.pad_token_id,
            },
            step_scores=step_scores,
            include_eos_baseline=include_eos_baseline,
            output_step_attributions=output_step_attributions,
            show_progress=False,
            batch_size=8,
        )

        # print(f"Attribution: {attribution}")
        
        explanations = []
        # Iterate over every step attribution output
        for step in attribution.step_attributions:
            # Extract target attribution scores from the step. If the score tensor
            # has more than two dimensions, take the norm across the last dimension.
            scores = step.target_attributions
            if scores.ndim > 2:
                scores = np.linalg.norm(scores, axis=-1)
            
            # Normalize scores so that they sum to 1.
            scores = scores / np.sum(scores)
            scores = np.squeeze(scores)  # Squeeze extra dimensions if any
            
            # Obtain the target token from this step. Here, we use the first target token.
            # target_token = step.target[0][0].token.replace("Ġ", "")
            target_token = step.target[0][0].token
            # Get the tokens from the prefix (the input tokens used for attribution)
            tokens = step.prefix[0]
            # Convert tokens to token ids then decode to text.
            token_ids = [token.id for token in tokens]
            tokens = [token.token for token in tokens]
            decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            # decoded_text = ' '.join(self.tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False))

            assert len(scores) == len(tokens), "Scores and tokens must have the same length."
            
            # Build an Explanation object for this step.
            explanation = Explanation(
                text=decoded_text,
                scores=scores,
                tokens=tokens,
                explainer=self.NAME,
                target=target_token
            )
            explanations.append(explanation)
        
        return explanations

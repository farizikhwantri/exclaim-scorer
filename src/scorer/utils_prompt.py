import random
from typing import Callable, Dict, List

task_prefix = {
    "gdpr": "Below is a Natural Language Inference (NLI) task for compliance detection in general data protection regulation domain",
    "hipaa": "Below is a Natural Language Inference (NLI) task for compliance detection in Health Insurance Portability and Accountability Act domain",
    "nli_gdpr": "Below is a Natural Language Inference (NLI) task for compliance detection in general data protection regulation domain.\n",
    "nli_medical": "Below is a Natural Language Inference (NLI) task for medical domain compliance detection.\n",
    "nli_cybersec": "Below is a Natural Language Inference (NLI) task for cybersecurity requirements compliance detection.\n",
    "nli_legal": "Below is a Natural Language Inference (NLI) task for legal document compliance detection.\n",
}

#  def build_prompt(premise, hypothesis):
def build_prompt(instance, task_prompt=None):
    # In-context examples can be appended as header examples.
    # Here we provide a simple instruction followed by the pair.
    premise = instance["premise"]
    hypothesis = instance["hypothesis"]
    if task_prompt is None:
        task_prompt = "Below is a Natural Language Inference (NLI) task for compliance detection in general data protection regulation domain.\n"
        task_prompt += "give answer in either 'entailment' or 'not entailment'\n\n"
    prompt = (
        f"{task_prompt}\n"
        "Example 1:\n"
        "Premise: The processor shall not engage a sub-processor without a prior specific or general written authorization of the controller.\n"
        "Hypothesis: Partner may not block or interfere with this monitoring.\n"
        "Answer: entailment\n\n"
        "Example 2:\n"
        "Premise: The DPA shall contain the duration of the processing.\n"
        "Hypothesis: Partner may not block or interfere with this monitoring.\n"
        "Answer: not entailment\n\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        "Answer:"
    )
    return prompt

def build_prompt_curried(pos_example: tuple, neg_example: tuple):
    """
    Returns a function that builds a prompt using fixed in-context examples.
    
    Parameters:
        pos_example: tuple of (premise, hypothesis) for an example labeled 'entailment'
        neg_example: tuple of (premise, hypothesis) for an example labeled 'not entailment'
        
    Returns:
        A function that accepts (premise, hypothesis) for an inference question.
    """
    def inner_prompt(instance, task_prompt=None):
        if task_prompt is None:
            task_prompt = "Below is a Natural Language Inference (NLI) task for compliance detection in general data protection regulation domain.\n"
            task_prompt += "give answer in either 'entailment' or 'not entailment'\n\n"
        premise = instance["premise"]
        hypothesis = instance["hypothesis"]
        prompt = (
            f"{task_prompt}"
            f"Example 1:\nPremise: {pos_example[0]}\n"
            f"Hypothesis: {pos_example[1]}\n"
            "Answer: entailment\n\n"
            f"Example 2:\nPremise: {neg_example[0]}\n"
            f"Hypothesis: {neg_example[1]}\n"
            "Answer: not entailment\n\n"
            f"Premise: {premise}\n"
            f"Hypothesis: {hypothesis}\n"
            "Answer:"
        )
        return prompt
    return inner_prompt

def build_shuffle_prompt_curried(pos_example: tuple, neg_example: tuple):
    """
    Returns a function that builds a prompt using fixed in-context examples,
    but shuffles the order of the examples.
    
    Parameters:
        pos_example: tuple of (premise, hypothesis) for an example labeled 'entailment'
        neg_example: tuple of (premise, hypothesis) for an example labeled 'not entailment'
        
    Returns:
        A function that accepts (premise, hypothesis) for an inference question.
    """
    # def inner_prompt(premise, hypothesis):
    def inner_prompt(instance, task_prompt=None):
        premise = instance["premise"]
        hypothesis = instance["hypothesis"]
        examples = [
            f"Premise: {pos_example[0]}\nHypothesis: {pos_example[1]}\nAnswer: entailment",
            f"Premise: {neg_example[0]}\nHypothesis: {neg_example[1]}\nAnswer: not entailment"
        ]
        random.shuffle(examples)
        prompt_examples = "\n\n".join([f"Example {i+1}:\n" + ex for i, ex in enumerate(examples)])
        if task_prompt is None:
            task_prompt = "Below is a Natural Language Inference (NLI) task for compliance detection in general data protection regulation domain.\n"
            task_prompt += "give answer in either 'entailment' or 'not entailment'\n\n"
        prompt = (
            f"{task_prompt}"
            f"{prompt_examples}\n\n"
            f"Premise: {premise}\n"
            f"Hypothesis: {hypothesis}\n"
            "Answer:"
        )
        return prompt
    return inner_prompt


# def zero_shot_prompt(premise, hypothesis):
def zero_shot_prompt(instance, task_prompt=None):
    premise = instance["premise"]
    hypothesis = instance["hypothesis"]
    # In-context examples can be appended as header examples.
    # Here we provide a simple instruction followed by the pair.
    if task_prompt is None:
        task_prompt = "Below is a Natural Language Inference (NLI) task for compliance detection in general data protection regulation domain.\n"
        task_prompt += "give answer in either 'entailment' or 'not entailment'\n\n"
    prompt = (
        f"{task_prompt}"
        "give answer in either 'entailment' or 'not entailment'\n\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        "Answer:"
    )
    return prompt

def make_multi_hop_as_in_context(premises: List[str], hypothesis: str, label: str=None, shuffle: bool=False) -> str:
    # Create the context for multi-hop reasoning
    context = []
    context.append(f"Premise: {premises[0]}\n")
    if len(premises) > 1:
        for i in range(1, len(premises)):
            # add the current pos premise as hypothesis to the previous premise
            context[-1] += f"Hypothesis: {premises[i]}\n"
            context[-1] += "Answer: entailment\n"
            if i < len(premises) - 1 or label:
                context.append(f"Premise: {premises[i]}\n")
    # context.append(f"Hypothesis: {hypothesis}\nAnswer: entailment\n")
    if label:
        context[-1] += f"Hypothesis: {hypothesis}\nAnswer: {label}\n"
    # if shuffle, shuffle the context
    if shuffle:
        random.shuffle(context)
    return context

def multi_hop_prompt(instance: Dict, separator: str = "|| ", task_prompt=None) -> str:
    premise = instance["premise"]
    hypothesis = instance["hypothesis"]
    # label = instance.get("label", "not entailment")
    # In-context examples can be appended as header examples.
    # Here we provide a simple instruction followed by the pair.
    # get number of hop
    num_hop = int(instance.get("hop", 1))
    premises = [p.strip() for p in premise.split(separator) if p.strip()]
    if num_hop > 1:
        assert len(premises) == num_hop, \
            f"instance {instance} has {premises}, {len(premises)} premises, \
              but num_hop is {num_hop}"
    # create the prompt by making the premise_1, premise_2, ..., premise_n as in_context examples 
    # in between them label as entailment
    if task_prompt is None:
        task_prompt = "Below is a Natural Language Inference (NLI) task for compliance detection in general data protection regulation domain.\n"
        task_prompt += "give answer in either 'entailment' or 'not entailment'\n\n"
    prompt = (
        f"{task_prompt}"
        "give answer in either 'entailment' or 'not entailment'\n\n"
        # f"Premise: {premises[0]}\n"
    )
    # if num_hop > 1:
    #     for i in range(1, num_hop):
    #         prompt += f"Premise {i+1}: {premises[i]}\n"
    #         prompt += "Answer: entailment\n"
    # prompt += f"Hypothesis: {hypothesis}\n"
    if len(premises) > 1:
        # add premise 0 
        prompt += f"Premise: {premises[0]}\n"
        for i in range(1, len(premises)):
            # make premises[i] as hypothesis to prev premise first then add Answer: entailment
            prompt += f"Hypothesis: {premises[i]}\n"
            prompt += "Answer: entailment\n"
            # make space for the next premise
            if i < len(premises) - 1:
                prompt += f"\nPremise: {premises[i]}\n"
            # prompt += "Answer: entailment\n"
    # prompt += f"Hypothesis: {hypothesis}\n"
    # prompt += f"Answer: {label}\n"
    # add premise 0 and the hypothesis to the prompt
    prompt += f"\nPremise: {premises[0]}\n"
    prompt += f"Hypothesis: {hypothesis}\n"
    prompt += "Answer:"
    return prompt

def multi_hop_prompt_curried(pos_example: tuple, neg_example: tuple, 
                             separator="|| ", shuffle_examples=True) -> Callable:
    """
    Returns a function that builds a multi-hop prompt using fixed in-context examples.
    
    Parameters:
        pos_example: tuple of (premise, hypothesis) for an example labeled 'entailment'
        neg_example: tuple of (premise, hypothesis) for an example labeled 'not entailment'
        
    Returns:
        A function that accepts (premise, hypothesis) for an inference question.
    """
    pos_example_premise, pos_example_hypothesis = pos_example
    neg_example_premise, neg_example_hypothesis = neg_example
    def iterate_premises(premise):
        # Split the premise by the separator and return a list of premises
        return [p.strip() for p in premise.split(separator) if p.strip()]
    
    # build the examples list
    pos_examples = []
    pos_premises = iterate_premises(pos_example_premise)
    pos_examples = make_multi_hop_as_in_context(pos_premises, 
                                                pos_example_hypothesis, 
                                                "entailment")

    # shuffle the positive examples
    # random.shuffle(pos_examples)

    neg_premises = iterate_premises(neg_example_premise)
    # add the first neg premise as premise:
    neg_examples = []
    neg_examples = make_multi_hop_as_in_context(neg_premises,
                                                neg_example_hypothesis, 
                                                "not entailment")

    # shuffle the negative examples
    # random.shuffle(neg_examples)

    # combine the examples
    examples = pos_examples + neg_examples
    # shuffle the examples
    if shuffle_examples:
        random.shuffle(examples)

    # make the examples into a single string of Example 1, Example 2, ...
    examples_str = "\n\n".join([f"Example {i+1}:\n" + "".join(ex) for i, ex in enumerate(examples)])
    # without example headers
    # examples_str = "\n".join(["".join(ex) for ex in examples])


    def inner_prompt(instance, task_prompt=None):
        # add in-context examples

        premise = instance["premise"]
        hypothesis = instance["hypothesis"]
        num_hop = int(instance.get("hop", 1))
        premises = [p.strip() for p in premise.split(separator) if p.strip()]
        if num_hop > 1:
            assert len(premises) == num_hop, \
                f"instance {instance} has {premises}, {len(premises)} premises, \
                  but num_hop is {num_hop}"
        # create the prompt by making the premise_1, premise_2, ..., premise_n as in_context examples 
        # in between them label as entailment
        # default_task = "Below is a Natural Language Inference (NLI) task for compliance detection in general data protection regulation domain.\n"
        if task_prompt is None:
            default_task = "Below is a Natural Language Inference (NLI) task for compliance detection in general data protection regulation domain.\n"
            task_prompt = default_task
        task_prompt += "give answer in either 'entailment' or 'not entailment'\n\n"
        if task_prompt is not None:
            prompt = task_prompt

        prompt = (
            f"{task_prompt}"
            "give answer in either 'entailment' or 'not entailment'\n\n"
            f"{examples_str}\n"
            "End of examples\n\n"
            # f"Premise: {premises[0]}\n"
        )
        
        if num_hop > 1 or len(premises) > 1:
            # add premise 0 
            prompt += f"Premise: {premises[0]}\n"
            for i in range(1, num_hop):
                # make premises[i] as hypothesis to prev premise first then add Answer: entailment
                prompt += f"Hypothesis: {premises[i]}\n"
                prompt += "Answer: entailment\n"
                # make space for the next premise
                if i < len(premises) - 1:
                    prompt += f"\nPremise: {premises[i]}\n"
                # prompt += "Answer: entailment\n"
        # prompt += f"Hypothesis: {hypothesis}\n"
        # prompt += "Answer: entailment\n"
        # add premise 0 and the hypothesis to the prompt
        prompt += f"\nPremise: {premises[0]}\n"
        prompt += f"Hypothesis: {hypothesis}\n"
        prompt += "Answer:"

        return prompt
    return inner_prompt

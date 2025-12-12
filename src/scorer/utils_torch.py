import time
import random
import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import data

from transformers import Trainer
from transformers import get_cosine_schedule_with_warmup

# Define a custom Trainer to incorporate knowledge distillation
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature=2.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs_student = model(**inputs)
        outputs_teacher = self.teacher_model(**inputs)
        student_loss = outputs_student.loss
        teacher_loss = outputs_teacher.loss
        distillation_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(outputs_student.logits / self.temperature, dim=-1),
            torch.nn.functional.softmax(outputs_teacher.logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        return (loss, outputs_student) if return_outputs else loss

# Function to access nested attributes
def get_nested_attr(obj, attr_path):
    attrs = attr_path.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj

# Supervised Contrastive Loss
def supervised_contrastive_loss(features, labels, temperature=0.07):
    features = F.normalize(features, p=2, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    similarity_matrix /= temperature
    batch_size = labels.size(0)
    mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    exp_sim = torch.exp(similarity_matrix)
    pos_sim = exp_sim * label_mask
    neg_sim = exp_sim * (~label_mask & ~mask)
    denominator = torch.sum(neg_sim, dim=1, keepdim=True)
    numerator = torch.sum(pos_sim, dim=1, keepdim=True)
    loss_per_sample = -torch.log(numerator / denominator)
    valid_samples = torch.sum(label_mask, dim=1) > 1
    return torch.sum(loss_per_sample[valid_samples]) / torch.sum(valid_samples)


def trainer_by_epochs(model: nn.Module, 
                      num_train_epochs: int, 
                      dataloader: data.DataLoader, 
                      optimizer: torch.optim.Optimizer, 
                      dataset: data.Dataset, 
                      device: str = 'cpu',
                      grad_accumulation_steps: int = 1,
                      warmup_steps: int = 100):
    total_training_steps = num_train_epochs * len(dataloader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )
    
    start_time = time.time()
    model.train()
    step = 0
    for epoch in range(num_train_epochs):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)

            batch_train = {
                "input_ids": batch["input_ids"].to(device=device),
                "attention_mask": batch["attention_mask"].to(device=device),
                "labels": batch["labels"].to(device=device),
            }
            # handle if model needs token_type_ids
            if "token_type_ids" in batch:
                batch_train["token_type_ids"] = batch["token_type_ids"].to(device=device)

            loss = model(**batch_train).loss
            loss = loss / grad_accumulation_steps
            loss.backward()

            if (step + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()    # update the learning rate scheduler
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.detach().float()
            if random.random() < 0.01:
                logging.info(f"batch loss: {loss.detach().float()}")
        
        # In case the number of steps is not an exact multiple of accumulation_steps
        if (step + 1) % grad_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        logging.info(f"Epoch {epoch + 1} - Averaged Loss: {total_loss / len(dataset)}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Completed training in {elapsed_time:.2f} seconds.")

    return model

def trainer_by_step(model: nn.Module, 
                    steps: int, 
                    optimizer: torch.optim.Optimizer, 
                    dataloader: data.DataLoader , 
                    device: str = 'cpu', 
                    logging_frequency=1000,
                    grad_accumulation_steps: int = 1,
                    warmup_steps: int = 100):
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=steps
    )

    start_time = time.time()
    model.train()
    step = 0
    while step < steps:
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)

            batch_train = {
                "input_ids": batch["input_ids"].to(device=device),
                "attention_mask": batch["attention_mask"].to(device=device),
                "labels": batch["labels"].to(device=device),
            }
            # handle if model needs token_type_ids
            if "token_type_ids" in batch:
                batch_train["token_type_ids"] = batch["token_type_ids"].to(device=device)

            loss = model(**batch_train).loss
            loss = loss / grad_accumulation_steps
            loss.backward()
            # optimizer.step()
            if (step + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()    # update the learning rate scheduler
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.detach().float()
            if random.random() < 0.01:
                logging.info(f"batch loss: {loss.detach().float()}")
            step += 1

            if step % logging_frequency == 0:
                logging.info(f"Step {step} - Averaged Loss: {total_loss / len(dataloader)}")

            if step >= steps:
                break
        
        if (step + 1) % grad_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Completed training in {elapsed_time:.2f} seconds.")

    return model

def get_max_memory():
    """Get the maximum memory available for loading models, checking CUDA, MPS, or CPU."""
    if torch.cuda.is_available():
        # CUDA is available
        free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
        max_memory = f'{free_in_GB-6}GB'
        n_gpus = torch.cuda.device_count()
        max_memory = {i: max_memory for i in range(n_gpus)}
        return {"CUDA": max_memory}
    
    elif torch.backends.mps.is_available():
        # MPS is available (Apple Silicon)
        # Currently, MPS does not have a direct method to get free memory
        print("MPS is available. Please provide the maximum memory available for MPS.")
        memory_in_GB = 16  # Example: replace with an actual estimation method if possible
        max_memory = f'{memory_in_GB-2}GB'
        return {"mps": max_memory}
    
    else:
        # Fallback for CPU-only systems
        import psutil
        memory_in_GB = int(psutil.virtual_memory().total / 1024**3)
        max_memory = f'{memory_in_GB-2}GB'
        return {"CPU": max_memory}

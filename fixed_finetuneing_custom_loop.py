#!/usr/bin/env python
# Fixed version of the Qwen Marathi fine-tuning script

# Cell 1: Imports
import numpy as np
import pandas as pd 
import os

# Cell 2-3: Install packages (commented out - run manually if needed)
# !pip install -q evaluate sacrebleu wandb

# Cell 4: Main imports
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding, get_scheduler
from datasets import load_dataset
import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers.optimization import AdafactorSchedule
from transformers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
import os
import math
from transformers import DataCollatorForSeq2Seq

# Cell 5: Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Set environment variable for multi-GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Cell 6: WandB login (commented out - set WANDB_API_KEY environment variable instead)
# from kaggle_secrets import UserSecretsClient
# secret = UserSecretsClient()
# os.environ["WANDB_API_KEY"] = secret.get_secret("WANDB_KEY") 
# wandb.login()

# Cell 7: Load model and tokenizer
model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Cell 10: Load dataset
ds = load_dataset("anujsahani01/English-Marathi")

# Cell 13: Slice dataset for quick experimentation
df_train = ds["train"].select(range(1000))
df_validation = ds["train"].select(range(1000, 1100))
df_test = ds["test"].select(range(100))

# Cell 14: Add instruction
def add_instruction(example):
    example["instruction"] = "Convert the English text into Marathi language."
    return example

df_train = df_train.map(add_instruction, batched=False)
df_validation = df_validation.map(add_instruction, batched=False)
df_test = df_test.map(add_instruction, batched=False)

# Cell 31: Tokenize function
def tokenize_function(examples):
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for instruction, english, marathi in zip(
            examples["instruction"], examples["english"], examples["marathi"]):

        prompt_text = f"{instruction} English: {english} Marathi:"
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids

        full_text = f"{prompt_text} {marathi}{tokenizer.eos_token}"
        tok = tokenizer(full_text, add_special_tokens=False)

        label_ids = [-100] * len(prompt_ids) + tok["input_ids"][len(prompt_ids):]

        batch_input_ids.append(tok["input_ids"])
        batch_attention_mask.append(tok["attention_mask"])
        batch_labels.append(label_ids)

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels,
    }

# Cell 32: Tokenize datasets
train_tokenised_ds = df_train.map(tokenize_function, batched=True, num_proc=4, remove_columns=df_train.column_names)
test_tokenised_ds = df_test.map(tokenize_function, batched=True, num_proc=4, remove_columns=df_test.column_names)
validation_tokenised_ds = df_validation.map(tokenize_function, batched=True, num_proc=4, remove_columns=df_validation.column_names)

# Cell 35: Set pad token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

# Cell 36: Set format
for split in (train_tokenised_ds, test_tokenised_ds, validation_tokenised_ds):
    split.set_format(type="torch")

# Cell 37: Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)

# Cell 39: Create data loaders
train_dataloader = DataLoader(
    train_tokenised_ds,
    batch_size=4,
    shuffle=True,
    collate_fn=data_collator,
    pin_memory=False
)

eval_dataloader = DataLoader(
    validation_tokenised_ds,
    batch_size=4,
    shuffle=False,
    collate_fn=data_collator,
    pin_memory=False
)

test_dataloader = DataLoader(
    test_tokenised_ds,
    batch_size=4,
    shuffle=False,
    collate_fn=data_collator,
    pin_memory=False
)

# Cell 40: Initialize Accelerator (FIXED - removed DataParallel)
accelerator = Accelerator(
    gradient_accumulation_steps=2,
    device_placement=True,
    mixed_precision="fp16",  
    log_with="wandb",
    project_dir="qwen_marathi"
)

# Don't use DataParallel - Accelerate handles multi-GPU automatically
print(f"Using Accelerate with {accelerator.num_processes} device(s)")

# Cell 41: Optimizer & scheduler
num_epochs = 1
per_device_train_batch_size = 2
per_device_eval_batch_size = 2
gradient_accumulation_steps = 2
learning_rate = 5e-5
weight_decay = 0.01
warmup_steps = 200
logging_steps = 500
eval_steps = 500
save_total_limit = 2
output_dir = "qwen_marathi"
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
max_train_steps = num_epochs * num_update_steps_per_epoch
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=max_train_steps,
)

# Cell 42: Prepare everything with Accelerate
model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler
)

# Cell 43: Print training setup
if accelerator.is_main_process:
    total_batch_size = per_device_train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    print(f"***** Running training *****")
    print(f"  Num examples              = {len(train_tokenised_ds)}")
    print(f"  Num Epochs                = {num_epochs}")
    print(f"  Instantaneous batch size  = {per_device_train_batch_size}")
    print(f"  Total train batch size    = {total_batch_size}")
    print(f"  Gradient Accumulation     = {gradient_accumulation_steps}")
    print(f"  Total optimization steps  = {max_train_steps}")
    print(f"  Number of devices         = {accelerator.num_processes}")

torch.cuda.empty_cache()

# Define helper functions
def evaluate_model(model, eval_dataloader, accelerator):
    """Evaluate the model and return average loss"""
    model.eval()
    total_eval_loss = 0.0
    eval_steps = 0
    
    if accelerator.is_main_process:
        eval_progress = tqdm(eval_dataloader, desc="Evaluating")
    else:
        eval_progress = eval_dataloader
    
    with torch.no_grad():
        for batch in eval_progress:
            outputs = model(**batch)
            loss = outputs.loss
            total_eval_loss += loss.detach().float()
            eval_steps += 1
    
    # Gather losses from all processes
    total_eval_loss = accelerator.gather(total_eval_loss).mean()
    eval_steps = accelerator.gather(torch.tensor(eval_steps, device=accelerator.device)).sum()
    
    avg_eval_loss = total_eval_loss / eval_steps
    return avg_eval_loss.item()

def save_checkpoint(model, tokenizer, accelerator, output_dir, checkpoint_name, saved_checkpoints, save_total_limit):
    checkpoint_dir = os.path.join(output_dir, checkpoint_name)    
    # Save model and tokenizer
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    
    if accelerator.is_main_process:
        # Save model
        unwrapped_model.save_pretrained(
            checkpoint_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            safe_serialization=True
        )
        # Save tokenizer
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Checkpoint saved to {checkpoint_dir}")
        saved_checkpoints.append(checkpoint_dir)        
        # Remove old checkpoints if limit exceeded
        if len(saved_checkpoints) > save_total_limit:
            oldest_checkpoint = saved_checkpoints.pop(0)
            if os.path.exists(oldest_checkpoint) and "best" not in oldest_checkpoint and "final" not in oldest_checkpoint:
                import shutil
                shutil.rmtree(oldest_checkpoint)
                print(f"Removed old checkpoint: {oldest_checkpoint}")

# Cell 44: Initialize WandB
if accelerator.is_main_process:
    wandb.init(
        project="qwen-marathi-finetuning",
        name="qwen-marathi-run2",
        config={
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "batch_size_per_device": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "weight_decay": weight_decay,
            "warmup_steps": warmup_steps,
            "optimizer": "AdamW",
            "lr_scheduler": "linear",
            "num_gpus": accelerator.num_processes,
        }
    )

# Set up training loop variables & progress bar
global_step = 0
total_loss = 0.0
best_eval_loss = float('inf')
saved_checkpoints = []

# Initialize progress bar (FIXED - added this)
if accelerator.is_main_process:
    progress_bar = tqdm(range(max_train_steps), desc="Training")

# Cell 45: Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.detach().float().item()
            total_loss += loss.detach().float().item()

        if accelerator.sync_gradients:
            global_step += 1

            # update tqdm
            if accelerator.is_main_process:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{lr_scheduler.get_last_lr()[0]:.2e}',
                    'step': global_step
                })

            # logging
            if global_step % logging_steps == 0:
                avg_loss = total_loss / global_step
                current_lr = lr_scheduler.get_last_lr()[0]
                if accelerator.is_main_process:
                    print(f"[Step {global_step}] train_loss={avg_loss:.4f}, lr={current_lr:.2e}")
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/learning_rate": current_lr,
                        "train/epoch": epoch + (step + 1) / len(train_dataloader),
                        "train/global_step": global_step
                    })

            # evaluation
            if global_step % eval_steps == 0:
                eval_loss = evaluate_model(model, eval_dataloader, accelerator)
                if accelerator.is_main_process:
                    print(f"[Step {global_step}] eval_loss={eval_loss:.4f}")
                    wandb.log({"eval/loss": eval_loss, "eval/global_step": global_step})
                    # save best
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        save_checkpoint(
                            model, tokenizer, accelerator, output_dir,
                            f"best-checkpoint-step{global_step}",
                            saved_checkpoints, save_total_limit
                        )
                model.train()

    # end of epoch
    if accelerator.is_main_process:
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} • avg_train_loss={avg_epoch_loss:.4f}")
        wandb.log({"train/epoch_loss": avg_epoch_loss, "train/epoch": epoch+1})

        # checkpoint at epoch end
        save_checkpoint(
            model, tokenizer, accelerator, output_dir,
            f"epoch-{epoch+1}",
            saved_checkpoints, save_total_limit
        )

# Final evaluation & cleanup
final_eval_loss = evaluate_model(model, eval_dataloader, accelerator)
if accelerator.is_main_process:
    print(f"Final evaluation • eval_loss={final_eval_loss:.4f}")
    wandb.log({"eval/final_loss": final_eval_loss})

    save_checkpoint(
        model, tokenizer, accelerator, output_dir,
        "final-checkpoint",
        saved_checkpoints, save_total_limit
    )

    progress_bar.close()
    wandb.finish()
    print("Training completed!")
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType
from datasets import load_dataset
from copy import deepcopy
from typing import List
import warnings
import random
import torch
import os
import dotenv
import wandb
from utils import apply_mapping


def finetune(model, tokenizer, random_seed=42):
    """
    Finetune the model on the SlimOrca dataset. The choice of parameters generally follow 
    those from Gromov et al. The Unreasonable Ineffectiveness of the Deeper Layers (2024).
    """

    def tokenize(examples):
        messages = examples["conversations"]
        text = [tokenizer.apply_chat_template(apply_mapping(m), tokenize=False, add_generation_prompt=False) for m in messages]
        return {"text": text}
    wandb.init(
            project="model_lobotomization",
    )

    dataset = load_dataset("Open-Orca/SlimOrca")
    dataset = dataset["train"].select(range(1000))

    dataset = dataset.map(tokenize, batched=True, batch_size=16).shuffle(seed=random_seed)

    sft_config = SFTConfig(
            dataset_text_field="text",
            max_seq_length=1024,
            learning_rate=3e-4,
            lr_scheduler_type="cosine",
            per_device_train_batch_size=1, # increase this if vram allows
            warmup_steps=0,
            max_steps=200,
            save_strategy="steps",
            save_steps=25,
            output_dir="output",
            logging_steps=1,
            logging_strategy="steps",
    )

    peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=2,
            lora_alpha=2,
            lora_dropout=0.05,
            target_modules=["gate_proj", "down_proj", "up_proj"]
    )

    trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=sft_config,
            peft_config=peft_config,
            tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save everything
    trainer.save_model("outputs/final_model")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'training_args': trainer.args,
        }, "outputs/training_state.pt")

    wandb.finish()

    return model

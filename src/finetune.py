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
from utils import apply_mapping

def finetune(model, tokenizer, random_seed):
    """
    Finetune the model on the SlimOrca dataset. The choice of parameters generally follow 
    those from Gromov et al. The Unreasonable Ineffectiveness of the Deeper Layers (2024).
    """
    model.gradient_checkpointing_enable()  # Trades computation for memory
    model.enable_input_require_grads()
    dataset = load_dataset("Open-Orca/SlimOrca")
    dataset = dataset["train"].take(1000)

    def tokenize(examples):
        messages = examples["conversations"]
        text = [tokenizer.apply_chat_template(apply_mapping(m), tokenize=False, add_generation_prompt=False) for m in messages]
        return {"text": text}

    dataset = dataset.map(tokenize, batched=True, batch_size=16).shuffle(seed=random_seed)

    sft_config = SFTConfig(
        dataset_text_field="text",
        max_seq_length=1024,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=1, # increase this if vram allows
        gradient_accumulation_steps=8,
        warmup_steps=0,
        max_steps=100,
        save_strategy="steps",
        save_steps=25,
        output_dir="output",
        logging_steps=1,
        logging_strategy="steps",
        optim="adamw_torch_fused",  # Use fused optimizer if available
        fp16=True,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=2,
        lora_alpha=2,
        lora_dropout=0.05,
        target_modules=["gate_proj", "down_proj", "up_proj"],
        inference_mode=False,
        )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )


    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trainer.train()

if __name__ == "__main__":
    main()

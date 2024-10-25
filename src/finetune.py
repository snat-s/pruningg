from transformers import AutoTokenizer
from datasets import load_dataset
import torch
import os
import dotenv
from utils import apply_mapping
import wandb
from unsloth import FastLanguageModel
import torch.nn as nn
from accelerate import Accelerator
dotenv.load_dotenv()

def finetune(model, tokenizer, random_seed):
    """
    Finetune a pre-pruned model using unsloth optimizations.
    Takes the actual model instance instead of loading from scratch.
    """
    wandb.init(
        project="model_lobotomization",
        config={
            "learning_rate": 3e-4,
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "max_steps": 10000,
            "lora_r": 2,
            "lora_alpha": 2,
            "lora_dropout": 0.05,
            "random_seed": random_seed,
            "max_seq_length": 1024,
        }
    )

    # Load and prepare the dataset
    dataset = load_dataset("Open-Orca/SlimOrca")
    dataset = dataset["train"].take(10_000)

    def tokenize(examples):
        messages = examples["conversations"]
        text = [tokenizer.apply_chat_template(apply_mapping(m), tokenize=False, add_generation_prompt=False) for m in messages]
        return {"text": text}

    dataset = dataset.map(tokenize, batched=True, batch_size=16).shuffle(seed=random_seed)

    # Apply unsloth's training optimizations to existing model
    model = FastLanguageModel.prepare_model(
        model,
        max_seq_length=1024,
        dtype=torch.bfloat16,
        device="auto",
        use_gradient_checkpointing=True
    )

    # Configure LoRA for the existing model
    lora_config = FastLanguageModel.get_peft_config(
        r=2,
        target_modules=["gate_proj", "down_proj", "up_proj"],
        lora_alpha=2,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=random_seed,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        peft_config=lora_config,
    )

    # Initialize Accelerator for distributed training
    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="bf16"
    )

    training_args = {
        "output_dir": "output",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 3e-4,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 0,
        "max_steps": 300,
        "logging_steps": 1,
        "save_steps": 25,
        "save_strategy": "steps",
        "bf16": True,
        "optim": "adamw_hf",
        "seed": random_seed,
    }

    trainer = FastLanguageModel.get_trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        max_seq_length=1024,
        dataset_text_field="text",
        packing=True,
        use_flash_attention=True,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trainer.train()
    trainer.save_model()
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'training_args': training_args,
    }, os.path.join(training_args["output_dir"], "training_state.pt"))
    
    wandb.finish()
    
    return model  

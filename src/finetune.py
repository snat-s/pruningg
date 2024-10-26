from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_sharegpt
from trl import SFTTrainer
from transformers import TrainingArguments
import wandb
import torch
from datasets import load_dataset

def finetune(model, tokenizer, random_seed):
    """
    Finetune a pre-pruned model using unsloth optimizations for Llama 3.2.
    """
    wandb.init(
        project="model_lobotomization",
        config={
            "learning_rate": 2e-4,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "max_steps": 60,
            "lora_r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0,
            "random_seed": random_seed,
            #"max_seq_length": 2048,
        }
    )

    # Load and prepare SlimOrca dataset
    dataset = load_dataset("Open-Orca/SlimOrca")
    dataset = dataset["train"].select(range(10_000))

    # Setup chat template for Llama 3.1/3.2 format
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    def apply_mapping(conversation):
        """Convert SlimOrca format to standard chat format"""
        return [
            {
                "role": "system" if msg["from"] == "system" else 
                        "assistant" if msg["from"] == "gpt" else 
                        "user" if msg["from"] == "human" else msg["from"],
                "content": msg["value"]
            }
            for msg in conversation
        ]

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(apply_mapping(convo), 
                                             tokenize=False, 
                                             add_generation_prompt=False) 
                for convo in convos]
        return {"text": texts}

    # Process dataset while keeping track of original columns
    original_columns = dataset.column_names
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        batch_size=16,
        remove_columns=original_columns
    ).shuffle(seed=random_seed)

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=random_seed,
    )

    # Setup the trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        #max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=random_seed,
            output_dir="outputs",
            report_to="none",
        ),
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

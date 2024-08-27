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

dotenv.load_dotenv()

def finetune(model, tokenizer, random_seed):
    """
    Finetune the model on the SlimOrca dataset. The choice of parameters generally follow 
    those from Gromov et al. The Unreasonable Ineffectiveness of the Deeper Layers (2024).
    """
    dataset = load_dataset("Open-Orca/SlimOrca")
    dataset = dataset["train"].select(range(1000))

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
        warmup_steps=0,
        max_steps=100,
        save_strategy="steps",
        save_steps=10,
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

    trainer.train()

    pass

def construct_ordered_model(model, order: List[int], use_cache: bool = True):
    if use_cache:
        original_layers = model.model.layers
        reordered_layers = torch.nn.ModuleList([deepcopy(original_layers[i]) for i in order])
    else:
        warnings.warn("Hugging Face's Attention implementation will not work with use_cache=False unless modified.", UserWarning)
        original_layers = model.model.layers
        reordered_layers = torch.nn.ModuleList([original_layers[i] for i in order])

    [setattr(layer.self_attn, 'layer_idx', i) for i, layer in enumerate(reordered_layers)]

    model.model.layers = reordered_layers
    return model

def block_to_layer(block_order: List[int], layers_per_block: int, num_layers: int) -> List[int]:
    layer_order = []
    for block in block_order:
        start_layer = block * layers_per_block
        end_layer = min((block + 1) * layers_per_block, num_layers)
        layer_order.extend(range(start_layer, end_layer))
    return layer_order

def main():

    # Configuration
    llm_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    num_blocks = 8
    use_cache = True
    device = (
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() else 
        "cpu"
    )
    random_seed = 42
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    tokenizer = AutoTokenizer.from_pretrained(llm_name, token=os.environ["HF_AUTH_TOKEN"])
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.bfloat16, trust_remote_code=True, token=os.environ["HF_AUTH_TOKEN"])
    model.to(device)
    model.config.pad_token_id = tokenizer.eos_token_id
    num_layers = len(model.model.layers)
    assert num_layers % num_blocks == 0
    layers_per_block = int(num_layers / num_blocks)

    # Test different orders
    block_order = [0,1,2,3,4,5,6,7]
    layer_order = block_to_layer(block_order, layers_per_block, num_layers)
    model = construct_ordered_model(model, layer_order, use_cache=use_cache)

    # todo: add an evaluation step here

    # Finetune
    finetune(model, tokenizer, random_seed)

    # todo: add an evaluation step here

    pass

if __name__ == "__main__":
    main()
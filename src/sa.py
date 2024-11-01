"""
    Simulated Annealing
"""
import random
import math
import torch
import copy
import gc
import json
import argparse
import datetime
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from finetune import finetune
from evaluate_model import evaluate_model

def create_initial_state(gene_length: int) -> List[int]:
    state = []
    
    # Parameters to control the shape of the probability curve
    edge_prob = 0.9  # Probability of keeping edge layers
    middle_prob = 0.6  # Probability of keeping middle layers
    
    for i in range(gene_length):
        # Convert position to range [-1, 1] where 0 is the middle
        x = (2 * i / (gene_length - 1)) - 1
        
        # Quadratic function that's higher at edges (-1 and 1) and lower in middle (0)
        a = (edge_prob - middle_prob)
        b = middle_prob
        prob = a * x * x + b
        
        # Create gene based on calculated probability
        state.append(1 if random.random() < prob else 0)
    
    # Force keeping first and last layers
    if gene_length > 2:
        state[0] = 1
        state[-1] = 1
    
    return state

def get_neighbor(state: List[int], temperature: float) -> List[int]:
    """
    Generate neighbor state with temperature-dependent perturbation.
    Higher temperatures allow more changes.
    """
    neighbor = state.copy()
    gene_length = len(state)
    
    # Number of bits to flip increases with temperature
    max_flips = max(1, int(temperature * 3))  # More flips at higher temperatures
    num_flips = random.randint(1, max_flips)
    
    # Calculate flip probabilities based on position
    edge_prob = 0.5 * (1 - temperature)  # Lower probability of flipping edges at low temperatures
    middle_prob = 0.8  # Higher probability of flipping middle layers
    
    flip_probs = []
    for i in range(gene_length):
        x = (2 * i / (gene_length - 1)) - 1
        a = (edge_prob - middle_prob)
        b = middle_prob
        prob = a * x * x + b
        flip_probs.append(prob)
    
    # Normalize probabilities
    total_prob = sum(flip_probs)
    flip_probs = [p / total_prob for p in flip_probs]
    
    # Select positions to flip
    positions = random.choices(range(gene_length), weights=flip_probs, k=num_flips)
    
    # Don't flip first or last layer if we have more than 2 layers
    if gene_length > 2:
        positions = [p if (p != 0 and p != gene_length-1) else random.randint(1, gene_length-2) 
                    for p in positions]
    
    # Flip selected bits
    for position in positions:
        neighbor[position] = 1 - neighbor[position]
    
    return neighbor

def acceptance_probability(old_cost: float, new_cost: float, temperature: float) -> float:
    """Calculate probability of accepting new state"""
    if new_cost > old_cost:
        return 1.0
    return math.exp((new_cost - old_cost) / temperature)

def remove_layers(model, bitmask: List[int]):
    """Remove layers according to bitmask"""
    pruned_model = copy.deepcopy(model)
    surviving_layers = torch.nn.ModuleList([
        layer for layer, bit in zip(pruned_model.model.layers, bitmask) if bit
    ])
    pruned_model.model.layers = surviving_layers
    pruned_model.config.num_hidden_layers = len(surviving_layers)
    for i, layer in enumerate(pruned_model.model.layers):
        layer.self_attn.layer_idx = i
    return pruned_model

def simulated_annealing(
    model,
    tokenizer,
    evaluate_model_fn,
    model_name: str,
    baseline_accuracy: float,
    initial_temp: float = 1.0,
    min_temp: float = 0.01,
    cooling_rate: float = 0.95,
    steps_per_temp: int = 5,
    output_file: Path = Path("sa_results.json")
) -> Tuple[List[int], float]:

    num_layers = len(model.model.layers)
    current_state = create_initial_state(num_layers)
    
    model.to('cpu')
    pruned_model = remove_layers(model, current_state)
    current_accuracy = evaluate_model_fn(pruned_model.to('cuda'), tokenizer)
    pruned_model.to('cpu')
    del pruned_model
    torch.cuda.empty_cache()
    gc.collect()
    
    best_state = current_state.copy()
    best_accuracy = current_accuracy
    temperature = initial_temp
    
    result_data = {
        "model_name": model_name,
        "baseline_accuracy": baseline_accuracy,
        "hyperparameters": {
            "initial_temp": initial_temp,
            "min_temp": min_temp,
            "cooling_rate": cooling_rate,
            "steps_per_temp": steps_per_temp
        },
        "temperature_stages": []
    }
    
    while temperature > min_temp:
        stage_data = {
            "temperature": temperature,
            "steps": []
        }
        
        for step in range(steps_per_temp):
            neighbor = get_neighbor(current_state, temperature)
            
            # Evaluate neighbor
            pruned_model = remove_layers(model, neighbor)
            neighbor_accuracy = evaluate_model_fn(pruned_model.to('cuda'), tokenizer)
            pruned_model.to('cpu')
            del pruned_model
            torch.cuda.empty_cache()
            gc.collect()
            
            # Calculate acceptance probability
            if acceptance_probability(current_accuracy, neighbor_accuracy, temperature) > random.random():
                current_state = neighbor
                current_accuracy = neighbor_accuracy
                
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_state = current_state.copy()
            
            step_data = {
                "step": step + 1,
                "accuracy": float(current_accuracy),
                "state": current_state.copy(),
                "total_active_layers": sum(current_state),
                "compression_ratio": round(sum(current_state) / num_layers, 2)
            }
            stage_data["steps"].append(step_data)
            
            print(f"Temperature {temperature:.4f}, Step {step + 1}: "
                  f"Accuracy = {current_accuracy:.4f} "
                  f"(Active layers: {sum(current_state)}/{num_layers}, "
                  f"{100 * sum(current_state) / num_layers:.1f}%)")
        
        result_data["temperature_stages"].append(stage_data)
        temperature *= cooling_rate
        
        # Save intermediate results
        result_data["best_configuration"] = {
            "state": best_state,
            "accuracy": float(best_accuracy),
            "active_layers": sum(best_state),
            "compression_ratio": round(sum(best_state) / num_layers, 2)
        }
        
        with open(output_file, "w") as f:
            json.dump(result_data, f, indent=2)
    
    return best_state, best_accuracy

def main(llm_name: str, output_file: Path = Path("sa_results.json")):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {llm_name}")
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        llm_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.to(device)
    
    baseline_accuracy = evaluate_model(model, tokenizer)
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    
    num_layers = len(model.model.layers)
    print(f"Model has {num_layers} layers")
    
    best_state, best_accuracy = simulated_annealing(
        model=model,
        tokenizer=tokenizer,
        evaluate_model_fn=evaluate_model,
        model_name=llm_name,
        baseline_accuracy=baseline_accuracy,
        initial_temp=1.0,
        min_temp=0.01,
        cooling_rate=0.95,
        steps_per_temp=5,
        output_file=output_file
    )
    
    print("\nApplying best configuration...")
    final_model = remove_layers(model, best_state)
    final_accuracy = evaluate_model(final_model.to('cuda'), tokenizer)
    print(f"Model accuracy with pruned configuration: {final_accuracy:.4f}")
    
    print("\nFinetuning pruned model...")
    finetune(final_model, tokenizer, 42)
    finetuned_accuracy = evaluate_model(final_model.to('cuda'), tokenizer)
    print(f"Model accuracy after finetune: {finetuned_accuracy:.4f}")
    
    with open(output_file, 'r') as f:
        results = json.load(f)
    
    results['finetuned_accuracy'] = float(finetuned_accuracy)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                      help='Name or path of the model')
    parser.add_argument('--output-file', type=str, default="sa_results.json",
                      help='Path to save results')
    args = parser.parse_args()
    
    output_file = Path(args.output_file)
    main(args.model, output_file)

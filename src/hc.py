"""
    Hill Climbing with Random Restarts and Detailed Layer Tracking.
"""
import random
import torch
import copy
import gc
import json
import argparse
import datetime
from pathlib import Path
from evaluate_model import evaluate_model
from typing import List, Tuple, Dict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from finetune import finetune

def create_individual(gene_length: int) -> List[int]:
    """
    Creates an individual with bias towards keeping outer layers and being more
    selective with middle layers. Uses a quadratic function to create a
    probability distribution that's higher at the edges and lower in the
    middle.
    """
    individual = []
    
    # Parameters to control the shape of the probability curve
    edge_prob = 0.9  # Probability of keeping edge layers
    middle_prob = 0.7  # Probability of keeping middle layers
    
    for i in range(gene_length):
        # Convert position to range [-1, 1] where 0 is the middle
        x = (2 * i / (gene_length - 1)) - 1
        
        # Quadratic function that's higher at edges (-1 and 1) and lower in middle (0)
        # p(x) = ax^2 + b where a and b are chosen to match our desired probabilities
        a = (edge_prob - middle_prob)
        b = middle_prob
        prob = a * x * x + b
        
        # Create gene based on calculated probability
        individual.append(1 if random.random() < prob else 0)
    
    # Force keeping first and last layers (optional, but often beneficial)
    if gene_length > 2:  # Only if we have more than 2 layers
        individual[0] = 1  # Keep first layer
        individual[-1] = 1  # Keep last layer
    
    return individual

def get_neighbor(state: List[int]) -> List[int]:
    """Get neighbor by flipping one random bit"""
    neighbor = state.copy()
    position = random.randint(0, len(state) - 1)
    neighbor[position] = 1 - neighbor[position]
    return neighbor

def get_neighbor_multi(state: List[int], num_flips: int = 1, min_flips: int = 1) -> List[int]:
    assert min_flips <= num_flips, "min_flips must be <= num_flips"
    assert min_flips >= 1, "min_flips must be at least 1"
    assert num_flips < len(state), "num_flips must be less than state length"
    
    neighbor = state.copy()
    gene_length = len(state)
    
    actual_flips = random.randint(min_flips, num_flips)
    valid_positions = list(range(1, gene_length - 1)) if gene_length > 2 else list(range(gene_length))
    positions_to_flip = random.sample(valid_positions, min(actual_flips, len(valid_positions)))
    
    for position in positions_to_flip:
        neighbor[position] = 1 - neighbor[position]
    
    return neighbor

def get_biased_neighbor(state: List[int]) -> List[int]:
    """
    Gets a neighbor state by flipping one bit, with bias towards keeping outer
    layers and being more selective with middle layers. Uses similar probability
    distribution as create_individual function.
    
    Args:
        state: Current state as a list of 0s and 1s
        
    Returns:
        List[int]: New neighbor state with one bit flipped
    """
    neighbor = state.copy()
    gene_length = len(state)
    
    # Parameters to control the shape of the probability curve
    edge_prob = 0.3  # Probability of flipping edge layers (lower = more stable)
    middle_prob = 0.7  # Probability of flipping middle layers (higher = more changeable)
    
    # Calculate flip probabilities for each position
    flip_probs = []
    for i in range(gene_length):
        # Convert position to range [-1, 1] where 0 is the middle
        x = (2 * i / (gene_length - 1)) - 1
        
        # Quadratic function that's lower at edges (-1 and 1) and higher in middle (0)
        # We invert the curve compared to create_individual since we want
        # higher probability of changes in the middle
        a = (edge_prob - middle_prob)
        b = middle_prob
        prob = a * x * x + b
        flip_probs.append(prob)
    
    # Normalize probabilities
    total_prob = sum(flip_probs)
    flip_probs = [p / total_prob for p in flip_probs]
    
    # Choose position to flip based on calculated probabilities
    position = random.choices(range(gene_length), weights=flip_probs)[0]
    
    # Don't flip first or last layer if we have more than 2 layers
    if gene_length > 2 and (position == 0 or position == gene_length - 1):
        # If we randomly selected an edge layer, choose a random middle layer instead
        position = random.randint(1, gene_length - 2)
    
    # Flip the chosen bit
    neighbor[position] = 1 - neighbor[position]
    
    return neighbor

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

def hill_climbing_with_restarts(
    model,
    tokenizer,
    evaluate_model_fn,
    model_name: str,
    baseline_accuracy: float,
    num_restarts: int = 2,
    steps_per_restart: int = 20,
    output_file: Path = Path("optimization_results.json")
) -> Tuple[List[int], float]:
    """Hill Climbing with Random Restarts"""

    num_layers = len(model.model.layers)
    best_state = None
    best_accuracy = float('-inf')
    
    result_data = {
        "model_name": model_name,
        "baseline_accuracy": baseline_accuracy,
        "runs": [],
        "best_configuration": None
    }

    for restart in range(num_restarts):
        print(f"\nRestart {restart + 1}/{num_restarts}")
        
        run_stats = {
            "steps": [],
            "initial_state": None,
            "initial_accuracy": None
        }
        
        # Start from random state
        current_state = create_individual(num_layers)
        run_stats["initial_state"] = current_state.copy()
        
        # Evaluate initial state
        model.to('cpu')
        pruned_model = remove_layers(model, current_state)
        current_accuracy = evaluate_model_fn(pruned_model.to('cuda'), tokenizer)
        run_stats["initial_accuracy"] = current_accuracy
        pruned_model.to('cpu')
        del pruned_model
        torch.cuda.empty_cache()
        gc.collect()
        
        # Hill Climbing
        for step in range(steps_per_restart):
            neighbor = get_neighbor_multi(current_state, 3)
            pruned_model = remove_layers(model, neighbor)
            neighbor_accuracy = evaluate_model_fn(pruned_model.to('cuda'), tokenizer)
            pruned_model.to('cpu')
            del pruned_model
            torch.cuda.empty_cache()
            gc.collect()
            
            if neighbor_accuracy > current_accuracy:
                current_state = neighbor
                current_accuracy = neighbor_accuracy
                
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_state = current_state.copy()
            
            step_stats = {
                "step": step + 1,
                "accuracy": current_accuracy,
                "state": neighbor.copy(),
                "total_active_layers": sum(current_state),
                "compression_ratio": round(sum(current_state) / num_layers, 2)
            }
            run_stats["steps"].append(step_stats)
            
            print(f"Step {step + 1}: Accuracy = {current_accuracy:.4f} "
                  f"(Active layers: {sum(current_state)}/{num_layers}, "
                  f"{sum(current_state) / num_layers}:.1f%)")
    
        result_data["runs"].append(run_stats)

    result_data["best_configuration"] = {
        "state": best_state,
        "accuracy": best_accuracy,
    }

    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=2)

    return best_state, best_accuracy

def load_best_configuration(file_path: Path) -> List[int]:
    with open(file_path, 'r') as f:
        results = json.load(f)
    
    best_config = results.get('best_configuration', {})
    if not best_config or 'state' not in best_config:
        raise ValueError("No valid best configuration found in the JSON file")
    
    return best_config['state']

def main(llm_name: str, output_file: Path = Path("optimization_results.json")):
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
    if output_file.exists():
        best_state = load_best_configuration(output_file)
    else:
        baseline_accuracy = evaluate_model(model, tokenizer)
        print(f"{baseline_accuracy=}")
        num_layers = len(model.model.layers)
        print(f"Model has {num_layers} layers")
        
        best_state, best_accuracy = hill_climbing_with_restarts(
            model=model,
            tokenizer=tokenizer,
            baseline_accuracy=baseline_accuracy,
            evaluate_model_fn=evaluate_model,
            model_name=llm_name,
            num_restarts=5,
            steps_per_restart=20,
            output_file=output_file
        )
        
    print("\nApplying best configuration...")
    final_model = remove_layers(model, best_state)
    final_accuracy = evaluate_model(final_model.to('cuda'), tokenizer)
    print(f"Model accuracy with pruned configuration: {final_accuracy:.4f}")
    print("Finetuning")
    finetune(final_model, tokenizer, 42)
    finetuned_accuracy = evaluate_model(final_model.to('cuda'), tokenizer)
    print(f"Model accuracy after finetune with pruned configuration: {finetuned_accuracy:.4f}")
    
    with open(output_file, 'r') as f:
        results = json.load(f) 

    results['finetuned_accuracy'] = finetuned_accuracy

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--output-file', type=str, default="optimization_results.json")
    args = parser.parse_args()
    
    output_file = Path(args.output_file)
    main(args.model, output_file)

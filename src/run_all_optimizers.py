"""
Runs hill climbing, simmulated annealing and genetic algorithms one after the other.
"""
import argparse
import json
import os
import torch
import copy
import gc
from pathlib import Path
from typing import List, Dict, Any, Callable
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

from hc import hill_climbing_with_restarts, remove_layers
from sa import simulated_annealing
from sga import genetic_algorithm
from evaluate_model import list_evaluate_model, evaluate_model
from finetune import finetune

def format_results(model_name: str, 
                  metaheuristic: str, 
                  initial_accuracies: Dict[str, Dict[str, Any]],
                  best_state: List[int],
                  best_accuracy: float,
                  num_layers: int,
                  finetuned_accuracies: Dict[str, Dict[str, Any]],
                  run_info: Dict[str, Any]) -> Dict[str, Any]:
    """Format results consistently across all metaheuristics."""
    return {
        "model_name": model_name,
        "metaheuristic": metaheuristic,
        "timestamp": datetime.now().isoformat(),
        "initial_accuracies": initial_accuracies,  # Save all metrics as they come
        "best_configuration": {
            "state": best_state,
            "accuracy": best_accuracy,
            "active_layers": sum(best_state),
            "total_layers": num_layers,
            "compression_ratio": round(sum(best_state) / num_layers, 2)
        },
        "finetuned_accuracies": finetuned_accuracies,  # Save all metrics as they come
        "run_info": run_info
    }

def use_metaheuristic(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_name: str,
    metaheuristic: str,
    initial_accuracies: Dict[str, float],
    task_names: List[str],
    output_dir: str = "results"
) -> None:
    """Run a specific metaheuristic and save results."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = Path(output_dir) / f"{model_name.replace('/', '_')}_{metaheuristic}.json"
    
    num_layers = len(model.model.layers)
    baseline_accuracy = sum(initial_accuracies.values()) / len(initial_accuracies)
    
    run_info = {}
    if metaheuristic == "hc":
        best_state, best_accuracy = hill_climbing_with_restarts(
            model=model,
            tokenizer=tokenizer,
            evaluate_model_fn=evaluate_model,
            model_name=model_name,
            baseline_accuracy=baseline_accuracy,
            num_restarts=50,
            steps_per_restart=10,
            output_file=output_file
        )
        run_info = {"num_restarts": 50, "steps_per_restart": 10}
        
    elif metaheuristic == "sa":
        best_state, best_accuracy = simulated_annealing(
            model=model,
            tokenizer=tokenizer,
            evaluate_model_fn=evaluate_model,
            model_name=model_name,
            baseline_accuracy=baseline_accuracy,
            initial_temp=1.0,
            min_temp=0.01,
            cooling_rate=0.95,
            steps_per_temp=5,
            output_file=output_file
        )
        run_info = {
            "initial_temp": 1.0,
            "min_temp": 0.01,
            "cooling_rate": 0.95,
            "steps_per_temp": 5
        }
        
    elif metaheuristic == "sga":
        best_state, best_accuracy, _ = genetic_algorithm(
            model=model,
            tokenizer=tokenizer,
            population_size=10,
            gene_length=num_layers,
            generations=100,
            mutation_rate=0.1,
            model_name=model_name,
            initial_accuracy=baseline_accuracy,
            tasks=task_names
        )
        run_info = {
            "population_size": 10,
            "generations": 100,
            "mutation_rate": 0.1
        }
    
    # Apply best configuration and finetune
    print(f"\nApplying best {metaheuristic} configuration...")
    final_model = remove_layers(model, best_state)
    final_model.to('cuda')
    
    print("\nFinetuning model...")
    finetune(final_model, tokenizer, 42)
    
    # Evaluate on all tasks after finetuning
    print("\nEvaluating finetuned model on all tasks...")
    finetuned_accuracies = list_evaluate_model(final_model, tokenizer, task_names)
    
    # Calculate average accuracy for logging (using whatever metrics are available)
    avg_finetuned_accuracy = 0
    metric_count = 0
    for task_metrics in finetuned_accuracies.values():
        for metric_name, value in task_metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                avg_finetuned_accuracy += value
                metric_count += 1
    avg_finetuned_accuracy = avg_finetuned_accuracy / metric_count if metric_count > 0 else 0
    print(f"Average finetuned accuracy across all metrics: {avg_finetuned_accuracy:.4f}")
    
    # Format and save results
    results = format_results(
        model_name=model_name,
        metaheuristic=metaheuristic,
        initial_accuracies=initial_accuracies,  # Now passing complete metrics
        best_state=best_state,
        best_accuracy=best_accuracy,
        num_layers=num_layers,
        finetuned_accuracies=finetuned_accuracies,  # Now passing complete metrics
        run_info=run_info
    )
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model
    save_dir = Path("models") / f"{model_name.replace('/', '_')}_{metaheuristic}"
    final_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # Cleanup
    final_model.to('cpu')
    del final_model
    torch.cuda.empty_cache()
    gc.collect()

def main(model_name: str):
    task_names = [
            'tinyArc', 'tinyGSM8k', 'tinyHellaswag', 'tinyMMLU', 'tinyTruthfulQA', 'tinyWinogrande'
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ["HF_AUTH_TOKEN"])
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=os.environ["HF_AUTH_TOKEN"]
    )
    model.to(device)
    
    num_layers = len(model.model.layers)
    print(f"Model has {num_layers} layers")
    
    # Initial evaluation on all tasks
    print("\nEvaluating initial model on all tasks...")
    initial_accuracies = list_evaluate_model(model, tokenizer, task_names)
    print("\nInitial accuracies:")
    for task, metrics in initial_accuracies.items():
        print(f"{task}:")
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                print(f"  {metric_name}: {value:.4f}")
    
    # Run all metaheuristics
    for metaheuristic in ["hc", "sa", "sga"]:
        print(f"\nRunning {metaheuristic.upper()}...")
        model_copy = copy.deepcopy(model)
        use_metaheuristic(
            model=model_copy,
            tokenizer=tokenizer,
            model_name=model_name,
            metaheuristic=metaheuristic,
            initial_accuracies=initial_accuracies,  # Pass complete metrics
            task_names=task_names
        )
        del model_copy
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run all metaheuristics for model layer optimization')
    parser.add_argument('--model', type=str, required=True,
                      help='Name or path of the model (e.g., "meta-llama/Llama-3.2-1B")')
    args = parser.parse_args()
    main(args.model)

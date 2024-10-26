"""
    Simple Genetic Algorithm (SGA)
"""
import random
import matplotlib.pyplot as plt
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import os
import copy
import gc
import json
import argparse
from finetune import finetune
from evaluate_model import evaluate_model

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
    middle_prob = 0.4  # Probability of keeping middle layers
    
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
    #if gene_length > 2:  # Only if we have more than 2 layers
    #    individual[0] = 1  # Keep first layer
    #    individual[-1] = 1  # Keep last layer
    
    return individual

def create_population(population_size: int, gene_length: int) -> List[List[int]]:
    return [create_individual(gene_length) for _ in range(population_size)]

# Recombination: 1-Point crossover
def crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutation: Bit flip
def mutate(individual: List[int], mutation_rate: float) -> List[int]:
    return [1 - gene if random.random() < mutation_rate else gene for gene in individual]

# Parent selection: Fitness proportional
def select_parents(population: List[List[int]], fitnesses: List[float]) -> Tuple[List[int], List[int]]:
    parent1 = random.choices(population, weights=fitnesses, k=1)[0]
    parent2 = random.choices(population, weights=fitnesses, k=1)[0]
    return parent1, parent2

# Survival selection: Generational
def create_next_generation(population: List[List[int]], fitnesses: List[float], mutation_rate: float) -> List[List[int]]:
    next_generation = []
    while len(next_generation) < len(population):
        parent1, parent2 = select_parents(population, fitnesses)
        child1, child2 = crossover(parent1, parent2)
        next_generation.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])
    return next_generation[:len(population)]

def remove_layers(model, bitmask: List[int]):
    pruned_model = copy.deepcopy(model)
    surviving_layers = torch.nn.ModuleList([layer for layer, bit in zip(pruned_model.model.layers, bitmask) if bit])
    pruned_model.model.layers = surviving_layers
    pruned_model.config.num_hidden_layers = len(surviving_layers)
    for i, layer in enumerate(pruned_model.model.layers):
        layer.self_attn.layer_idx = i
    return pruned_model

def generate_completion(model, tokenizer, prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def genetic_algorithm(model, tokenizer, population_size: int, gene_length: int, generations: int, mutation_rate: float, model_name: str, initial_accuracy: float, tasks: List[str]) -> Tuple[List[int], float]:
    population = create_population(population_size, gene_length)
    best_accuracy = 0
    best_individual = None
    results = {
        "model_name": model_name,
        "initial_accuracy": initial_accuracy,
        "tasks": tasks,
        "final_accuracy": None,  # Will be updated at the end
        "generations": [],
        "hyperparameters": {
            "population_size": population_size,
            "gene_length": gene_length,
            "generations": generations,
            "mutation_rate": mutation_rate
        }
    }

    for gen in tqdm(range(generations), desc="Generations"):
        fitnesses = []
        generation_results = {
            "generation_number": gen + 1,
            "best_accuracy": None,
            "best_individual": None,
            "individuals": []
        }

        for i, individual in enumerate(population):
            torch.cuda.empty_cache()
            gc.collect()

            pruned_model = remove_layers(model, individual)
            pruned_model.to('cuda')
            accuracy = evaluate_model(pruned_model, tokenizer)
            fitnesses.append(accuracy)
            active_layers = sum(individual)
            
            individual_result = {
                "active_layers": active_layers,
                "total_layers": gene_length,
                "active_percentage": round(active_layers/gene_length * 100, 2),
                "accuracy": round(accuracy, 4),
                "individual": individual,
            }
            generation_results["individuals"].append(individual_result)
            
            print(f"{i}th model with {active_layers}/{gene_length} layers ({active_layers/gene_length*100:.1f}%): Accuracy = {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_individual = individual
                generation_results["best_accuracy"] = round(accuracy, 4)
                generation_results["best_individual"] = individual

            pruned_model.to('cpu')
            del pruned_model
            torch.cuda.empty_cache()
            gc.collect()

        population = create_next_generation(population, fitnesses, mutation_rate)
        print(f"Generation {gen+1}/{generations}: Best accuracy = {best_accuracy:.4f}")
        
        results["generations"].append(generation_results)
        
        with open("ga_results.json", "w") as f:
            json.dump(results, f, indent=2)

    results["final_accuracy"] = round(best_accuracy, 4)
    with open("ga_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return best_individual, best_accuracy, results

def plot_probability_distribution(gene_length: int):
    edge_prob = 0.9
    middle_prob = 0.4
    x = [(2 * i / (gene_length - 1)) - 1 for i in range(gene_length)]
    a = (edge_prob - middle_prob)
    b = middle_prob
    probs = [a * xi * xi + b for xi in x]
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(gene_length), probs, 'b-')
    plt.title('Probability Distribution for Layer Selection')
    plt.xlabel('Layer Index')
    plt.ylabel('Probability of Keeping Layer')
    plt.grid(True)
    plt.savefig('layer_probability_distribution.png')
    plt.close()

def main(llm_name="Qwen/Qwen2-0.5B-Instruct"):
    tasks = ["tinyMMLU"]
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(llm_name, token=os.environ["HF_AUTH_TOKEN"])
    tokenizer.pad_token = tokenizer.eos_token
    print(llm_name)
    model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.bfloat16, trust_remote_code=True, token=os.environ["HF_AUTH_TOKEN"])
    model.to(device)

    num_layers = len(model.model.layers)
    print(f"Model has {num_layers} layers")
    plot_probability_distribution(num_layers)

    if os.path.exists("ga_results.json"):
        print("Found existing results, skipping genetic algorithm...")
        with open("ga_results.json", "r") as f:
            results = json.load(f)

        best_accuracy = float('-inf')
        best_individual = None
        best_generation = None
        for generation_num, generation in enumerate(results['generations']):
            for individual in generation['individuals']:
                if individual['accuracy'] > best_accuracy:
                    best_accuracy = individual['accuracy']
                    best_individual = individual
                    best_generation = generation_num
                    best_binary_config = generation['best_individual']
        best_gene = best_binary_config
    else:
        print("No existing results found, running genetic algorithm...")
        print("Initial model evaluation:")
        initial_accuracy = evaluate_model(model, tokenizer, task_names=["tinyMMLU", "tinyHellaswag"])
        model.to('cpu')
        print(f"Initial accuracy: {initial_accuracy:.4f}")

        # Adjust these parameters as needed
        population_size = 5
        generations = 15
        mutation_rate = 0.05

        best_gene, best_accuracy, results = genetic_algorithm(
            model=model,
            tokenizer=tokenizer,
            population_size=population_size,
            gene_length=num_layers,
            generations=generations,
            mutation_rate=mutation_rate,
            model_name=llm_name,
            initial_accuracy=initial_accuracy,
            tasks=tasks
        )

    print("\nApplying best solution and evaluating...")
    print(f"Layers kept: {sum(best_gene)}/{num_layers} ({sum(best_gene)/num_layers*100:.1f}%)")
    final_model = remove_layers(model.to('cpu'), best_gene)

    final_model.to(device)
    final_accuracy = evaluate_model(final_model, tokenizer, task_names=["tinyMMLU", "tinyHellaswag"])
    print(f"Final model accuracy: {final_accuracy:.4f}")
    print(f"Layers removed: {num_layers - sum(best_gene)}")
    
    print("\nFinetuning model...")
    finetune(final_model, tokenizer, 42)
    final_accuracy = evaluate_model(final_model, tokenizer, task_names=["tinyMMLU", "tinyHellaswag"])
    print("Accuracy after finetune:", final_accuracy)

    results["accuracy_after_finetune"] = round(float(final_accuracy_after_finetune), 4)
    with open("ga_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run genetic algorithm for model layer optimization')
    parser.add_argument('--model', type=str, required=True,
                      help='Name or path of the model (e.g., "meta-llama/Llama-3.2-1B")')
    args = parser.parse_args()
    main(args.model)


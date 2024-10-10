"""
Simple Genetic Algorithm (SGA)
"""

import random
import matplotlib.pyplot as plt
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm
import os
import copy
import gc
import json

#login(token=os.environ["HF_AUTH_TOKEN"])

# Representation: Bit-strings
def create_individual(gene_length: int) -> List[int]:
    return [random.randint(0, 1) for _ in range(gene_length)]

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

def evaluate_model(model, tokenizer, task="mmlu", subset="high_school_computer_science"):
    dataset = load_dataset("cais/mmlu", subset, split="test")

    def format_example(question, choices):
        # Few shot for model adherance
        few_shot_prompt = (
                    "Example 1: What is the capital of France?\nA. Berlin\nB. Madrid\nC. Paris\nD. Rome\nAnswer: C\n\n"
                    "Example 2: What is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer: B\n\n"
        )
        prompt = few_shot_prompt + question + "\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:"
        return prompt

    def evaluate_batch(batch):
        prompts = [format_example(q, c) for q, c in zip(batch['question'], batch['choices'])]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=2)
        # TODO: Decode the answers and parse it, because only 2 tokens is really small.
        generated_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [ans.strip()[-1] for ans in generated_answers]

    all_predictions = []
    all_labels = []

    for i in range(0, len(dataset), 1):
        batch = dataset[i:i+1]
        predictions = evaluate_batch(batch)
        all_predictions.extend(predictions)
        all_labels.extend([chr(65 + ans) for ans in batch['answer']])
    correct = sum(pred == label for pred, label in zip(all_predictions, all_labels))
    accuracy = correct / len(all_predictions)

    return accuracy

def genetic_algorithm(model, tokenizer, population_size: int, gene_length: int, generations: int, mutation_rate: float) -> Tuple[List[int], float]:
    population = create_population(population_size, gene_length)
    best_accuracy = 0
    best_individual = None
    results = []

    for gen in tqdm(range(generations), desc="Generations"):
        fitnesses = []
        for individual in population:
            torch.cuda.empty_cache()
            gc.collect()

            pruned_model = remove_layers(model, individual)
            pruned_model.to('cuda')
            accuracy = evaluate_model(pruned_model, tokenizer)
            fitnesses.append(accuracy)
            print(f"Model with {sum(individual)} layers: Accuracy = {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_individual = individual

            pruned_model.to('cpu')
            del pruned_model
            torch.cuda.empty_cache()
            gc.collect()

        population = create_next_generation(population, fitnesses, mutation_rate)
        print(f"Generation {gen+1}/{generations}: Best accuracy = {best_accuracy:.4f}")
        results.append({
            "generation": gen + 1,
            "best_accuracy": best_accuracy,
            "best_individual": best_individual
        })
        with open("ga_results.json", "w") as f:
            json.dump(results, f)

    return best_individual, best_accuracy, results

def generate_completion(model, tokenizer, prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    llm_name = "Qwen/Qwen2-0.5B-Instruct" # starting incredibly small because i am gpu poor 
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(llm_name, token=os.environ["HF_AUTH_TOKEN"])
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.bfloat16, trust_remote_code=True, token=os.environ["HF_AUTH_TOKEN"])
    model.to(device)
    model.config.pad_token_id = tokenizer.eos_token_id

    num_layers = len(model.model.layers)

    print("Initial model evaluation:")
    initial_accuracy = evaluate_model(model, tokenizer)
    model.to('cpu')
    print(f"Initial accuracy: {initial_accuracy:.4f}")

    # Adjust these parameters as needed
    population_size = 20
    generations = 100
    mutation_rate = 0.01

    best_gene, best_accuracy, results = genetic_algorithm(
        model=model,
        tokenizer=tokenizer,
        population_size=population_size,
        gene_length=num_layers,
        generations=generations,
        mutation_rate=mutation_rate
    )

    print(f"\nBest solution:")
    print(f"Gene: {best_gene}")
    print(f"Accuracy: {best_accuracy:.4f}")
    print(f"Layers kept: {sum(best_gene)}/{num_layers}")

    final_model = remove_layers(model, best_gene)
    final_accuracy = evaluate_model(final_model.to('cuda'), tokenizer)
    print(f"Final model accuracy: {final_accuracy:.4f}")
    print(f"Layers removed: {num_layers - sum(best_gene)}")
    prompt = "Answer correctly. Which one is heavier? A kilogram of steel or a kilogram of feathers? Answer: "
    print("After removing layers:")
    print(generate_completion(final_model.to('cuda'), tokenizer, prompt))


    # ploting
    generations = [r["generation"] for r in results]
    accuracies = [r["best_accuracy"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(generations, accuracies)
    plt.title(f"Best Accuracy Over Generations for {llm_name}")
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("accuracy_over_time.png")
    plt.show()

if __name__ == "__main__":
    main()

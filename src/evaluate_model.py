from lm_eval import evaluator, tasks, utils
import lm_eval
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def print_eval_results(results):
    for task_name in results['results']:
        accuracy = results['results'][task_name]['acc,none']
        std_err = results['results'][task_name]['acc_stderr,none']
        
        # Handle aggregate categories differently than individual subtasks
        if task_name in results['n-samples']:
            n_samples = results['n-samples'][task_name]
            print(f"{task_name} Results (n={n_samples}):")
        else:
            print(f"{task_name} Results:")
            
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Accuracy StdErr: ±{std_err:.3f}")

def list_evaluate_model(model, tokenizer, task_names):
    "eval on a list of tasks instead of a single task and return everything"
    wrapped_model = lm_eval.models.huggingface.HFLM(pretrained=model, tokenizer=tokenizer)
    results = lm_eval.simple_evaluate(
        model=wrapped_model,
        tasks=task_names,
    )
    accuracies = {}

    for task_name in task_names:
        accuracies[task_name] = results['results'][task_name]

    return accuracies

def evaluate_model(model, tokenizer, task_names=["tinyMMLU"], limit=10):
    wrapped_model = lm_eval.models.huggingface.HFLM(pretrained=model, tokenizer=tokenizer)
    
    results = lm_eval.simple_evaluate(
        model=wrapped_model,
        tasks=task_names,
    )
    accuracies = {}
    for task_name in task_names:
        accuracies[task_name] = results['results'][task_name]['acc_norm,none']

    avg_accuracy = sum(accuracies.values()) / len(accuracies)
    for task, acc in accuracies.items():
        print(f"{task}: {acc:.4f}")
    print(f"\nAverage accuracy across all tasks: {avg_accuracy:.4f}")
    
    return avg_accuracy

if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-0.5B"#"meta-llama/Llama-3.2-1B" 
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    acc = evaluate_model(model, tokenizer)
    print(acc)

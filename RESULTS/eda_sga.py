import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def calculate_generation_stats(generation):
    """Calculate average and best accuracy for a generation"""
    if 'individuals' not in generation:
        return None, None
    
    accuracies = [ind['accuracy'] for ind in generation['individuals'] 
                 if 'accuracy' in ind and ind['accuracy'] is not None]
    
    if not accuracies:
        return None, None
        
    return np.mean(accuracies), np.max(accuracies)

def create_visualizations(data):
    # Create a figure with GridSpec for better layout control - now 3x2
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 2)
    
    # 1. Plot accuracy comparison (initial vs final vs after_finetune)
    ax1 = fig.add_subplot(gs[0, 0])
    accuracies = ['Initial', 'Final', 'After Finetune']
    values = [data['initial_accuracy'], data['final_accuracy'], data['accuracy_after_finetune']]
    ax1.bar(accuracies, values)
    ax1.set_title('Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    
    # 2. Plot best individual's layer activation pattern
    ax2 = fig.add_subplot(gs[0, 1])
    best_individual = data['generations'][0]['best_individual']
    ax2.imshow([best_individual], cmap='Blues', aspect='auto')
    ax2.set_title('Best Individual Layer Activation Pattern')
    ax2.set_xlabel('Layer Index')
    ax2.set_yticks([])
    
    # 3. Plot distribution of active layers percentage
    ax3 = fig.add_subplot(gs[1, 0])
    active_percentages = []
    for gen in data['generations']:
        if 'individuals' in gen:
            for ind in gen['individuals']:
                if 'active_percentage' in ind:
                    active_percentages.append(ind['active_percentage'])
    
    ax3.hist(active_percentages, bins=20)
    ax3.set_title('Distribution of Active Layers Percentage')
    ax3.set_xlabel('Active Layers (%)')
    ax3.set_ylabel('Count')
    
    # 4. Plot accuracy vs active layers percentage
    ax4 = fig.add_subplot(gs[1, 1])
    accuracies = []
    active_percentages_scatter = []
    for gen in data['generations']:
        if 'individuals' in gen:
            for ind in gen['individuals']:
                if 'accuracy' in ind and 'active_percentage' in ind:
                    accuracies.append(ind['accuracy'])
                    active_percentages_scatter.append(ind['active_percentage'])
    
    ax4.scatter(active_percentages_scatter, accuracies, alpha=0.5)
    ax4.set_title('Accuracy vs Active Layers Percentage')
    ax4.set_xlabel('Active Layers (%)')
    ax4.set_ylabel('Accuracy')
    
    # 5. Plot average and best accuracy over generations
    ax5 = fig.add_subplot(gs[2, :])
    generation_numbers = []
    avg_accuracies = []
    best_accuracies = []
    
    for i, gen in enumerate(data['generations']):
        avg_acc, best_acc = calculate_generation_stats(gen)
        if avg_acc is not None and best_acc is not None:
            generation_numbers.append(i + 1)
            avg_accuracies.append(avg_acc)
            best_accuracies.append(best_acc)
    
    # Plot the evolution of average and best accuracy
    if generation_numbers and avg_accuracies:
        # Plot average accuracy
        ax5.plot(generation_numbers, avg_accuracies, marker='o', linestyle='-', linewidth=2, markersize=6,
                label='Population Average', color='blue', alpha=0.7)
        
        # Plot best accuracy
        ax5.plot(generation_numbers, best_accuracies, marker='s', linestyle='-', linewidth=2, markersize=6,
                label='Best Individual', color='orange', alpha=0.7)
        
        ax5.set_title('Population Performance Over Generations')
        ax5.set_xlabel('Generation')
        ax5.set_ylabel('Accuracy')
        
        # Add horizontal lines for reference
        ax5.axhline(y=data['initial_accuracy'], color='r', linestyle='--', alpha=0.5, 
                    label=f'Initial Accuracy: {data["initial_accuracy"]:.4f}')
        ax5.axhline(y=data['accuracy_after_finetune'], color='g', linestyle='--', alpha=0.5,
                    label=f'After Finetune: {data["accuracy_after_finetune"]:.4f}')
        ax5.legend()
        
        # Add grid for better readability
        ax5.grid(True, alpha=0.3)
    
    # Adjust layout and add metadata
    plt.suptitle(f'Genetic Algorithm Analysis - {data["model_name"]}', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('ga_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load and visualize the data
    data = load_data('ga_results_QWEN_32B.json')
    create_visualizations(data)
    
    # Print some summary statistics
    print(f"Model: {data['model_name']}")
    print(f"Initial Accuracy: {data['initial_accuracy']:.4f}")
    print(f"Final Accuracy: {data['final_accuracy']:.4f}")
    print(f"Accuracy After Finetune: {data['accuracy_after_finetune']:.4f}")
    print("\nHyperparameters:")
    for key, value in data['hyperparameters'].items():
        print(f"- {key}: {value}")

if __name__ == "__main__":
    main()

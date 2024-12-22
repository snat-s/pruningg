import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def calculate_temperature_stats(temperature_stage):
    """Calculate average and best accuracy for a temperature stage"""
    accuracies = [step['accuracy'] for step in temperature_stage['steps']]
    return np.mean(accuracies), np.max(accuracies)

def create_visualizations(data):
    # Create a figure with GridSpec for better layout control
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 2)
    
    # 1. Plot baseline vs best accuracy comparison
    ax1 = fig.add_subplot(gs[0, 0])
    accuracies = ['Baseline', 'Best Configuration', 'Finetuned']
    values = [data['baseline_accuracy'], data['best_configuration']['accuracy'], 
              data['finetuned_accuracy']]
    ax1.bar(accuracies, values)
    ax1.set_title('Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    
    # 2. Plot best configuration's layer activation pattern
    ax2 = fig.add_subplot(gs[0, 1])
    best_state = data['best_configuration']['state']
    ax2.imshow([best_state], cmap='Blues', aspect='auto')
    ax2.set_title('Best Configuration Layer Activation Pattern')
    ax2.set_xlabel('Layer Index')
    ax2.set_yticks([])
    
    # 3. Plot temperature decay
    ax3 = fig.add_subplot(gs[1, 0])
    temperatures = [stage['temperature'] for stage in data['temperature_stages']]
    ax3.plot(temperatures, marker='o')
    ax3.set_title('Temperature Decay')
    ax3.set_xlabel('Stage')
    ax3.set_ylabel('Temperature')
    ax3.set_yscale('log')  # Using log scale for better visualization
    
    # 4. Plot compression ratio vs accuracy
    ax4 = fig.add_subplot(gs[1, 1])
    accuracies = []
    compression_ratios = []
    for stage in data['temperature_stages']:
        for step in stage['steps']:
            accuracies.append(step['accuracy'])
            compression_ratios.append(step['compression_ratio'])
    
    ax4.scatter(compression_ratios, accuracies, alpha=0.5)
    ax4.set_title('Accuracy vs Compression Ratio')
    ax4.set_xlabel('Compression Ratio')
    ax4.set_ylabel('Accuracy')
    
    # 5. Plot accuracy evolution over temperature stages
    ax5 = fig.add_subplot(gs[2, :])
    avg_accuracies = []
    best_accuracies = []
    temperatures = []
    
    for stage in data['temperature_stages']:
        avg_acc, best_acc = calculate_temperature_stats(stage)
        avg_accuracies.append(avg_acc)
        best_accuracies.append(best_acc)
        temperatures.append(stage['temperature'])
    
    # Create temperature stage numbers for x-axis
    stage_numbers = list(range(1, len(temperatures) + 1))
    
    # Plot average and best accuracies
    ax5.plot(stage_numbers, avg_accuracies, marker='o', linestyle='-', 
             label='Average Accuracy', color='blue', alpha=0.7)
    ax5.plot(stage_numbers, best_accuracies, marker='s', linestyle='-',
             label='Best Accuracy', color='orange', alpha=0.7)
    
    # Add reference lines
    ax5.axhline(y=data['baseline_accuracy'], color='r', linestyle='--', alpha=0.5,
                label=f'Baseline Accuracy: {data["baseline_accuracy"]:.4f}')
    ax5.axhline(y=data['finetuned_accuracy'], color='g', linestyle='--', alpha=0.5,
                label=f'Finetuned Accuracy: {data["finetuned_accuracy"]:.4f}')
    
    ax5.set_title('Accuracy Evolution Over Temperature Stages')
    ax5.set_xlabel('Temperature Stage')
    ax5.set_ylabel('Accuracy')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Add overall title and adjust layout
    plt.suptitle(f'Simulated Annealing Analysis - {data["model_name"]}', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('sa_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load and visualize the data
    data = load_data('hermes-llama-3.1-sa.json')
    create_visualizations(data)
    
    # Print summary statistics
    print(f"Model: {data['model_name']}")
    print(f"Baseline Accuracy: {data['baseline_accuracy']:.4f}")
    print(f"Best Configuration Accuracy: {data['best_configuration']['accuracy']:.4f}")
    print(f"Finetuned Accuracy: {data['finetuned_accuracy']:.4f}")
    print(f"\nBest Configuration Stats:")
    print(f"- Active Layers: {data['best_configuration']['active_layers']}")
    print(f"- Compression Ratio: {data['best_configuration']['compression_ratio']:.2f}")
    print("\nHyperparameters:")
    for key, value in data['hyperparameters'].items():
        print(f"- {key}: {value}")

if __name__ == "__main__":
    main()

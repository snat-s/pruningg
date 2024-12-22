import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def load_data(filename):
    """Load and parse JSON data"""
    with open(filename, 'r') as f:
        return json.load(f)

def create_visualizations(data):
    # Create figure with GridSpec for better layout control
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 2)
    
    # 1. Plot baseline accuracy reference
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axhline(y=data['baseline_accuracy'], color='r', linestyle='--', 
                label=f'Baseline Accuracy: {data["baseline_accuracy"]:.4f}')
    ax1.set_title('Baseline Accuracy Reference')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Plot layer activation patterns for best configuration found
    ax2 = fig.add_subplot(gs[0, 1])
    if data['runs'][0]['steps']:
        final_state = data['runs'][0]['steps'][-1]['state']
        ax2.imshow([final_state], cmap='Blues', aspect='auto')
        ax2.set_title('Final Layer Activation Pattern')
        ax2.set_xlabel('Layer Index')
        ax2.set_yticks([])
        
        # Add text annotations for active (1) and inactive (0) layers
        for i, val in enumerate(final_state):
            ax2.text(i, 0, str(val), ha='center', va='center')
    
    # 3. Plot distribution of compression ratios across all runs and steps
    ax3 = fig.add_subplot(gs[1, 0])
    compression_ratios = []
    for run in data['runs']:
        for step in run['steps']:
            compression_ratios.append(step['compression_ratio'])
    
    ax3.hist(compression_ratios, bins=20, alpha=0.7)
    ax3.set_title('Distribution of Compression Ratios')
    ax3.set_xlabel('Compression Ratio')
    ax3.set_ylabel('Count')
    
    # 4. Plot accuracy vs compression ratio scatter
    ax4 = fig.add_subplot(gs[1, 1])
    accuracies = []
    compression_ratios_scatter = []
    for run in data['runs']:
        for step in run['steps']:
            accuracies.append(step['accuracy'])
            compression_ratios_scatter.append(step['compression_ratio'])
    
    ax4.scatter(compression_ratios_scatter, accuracies, alpha=0.5)
    ax4.set_title('Accuracy vs Compression Ratio')
    ax4.set_xlabel('Compression Ratio')
    ax4.set_ylabel('Accuracy')
    
    # 5. Plot accuracy over steps for all runs (main visualization)
    ax5 = fig.add_subplot(gs[2, :])
    
    # Plot each run with a different color
    colors = plt.cm.rainbow(np.linspace(0, 1, len(data['runs'])))
    
    for run_idx, run in enumerate(data['runs']):
        steps = [step['step'] for step in run['steps']]
        accuracies = [step['accuracy'] for step in run['steps']]
        
        ax5.plot(steps, accuracies, marker='o', linestyle='-', linewidth=2, 
                markersize=6, color=colors[run_idx], alpha=0.7,
                label=f'Run {run_idx + 1}')
    
    # Add baseline accuracy reference
    ax5.axhline(y=data['baseline_accuracy'], color='r', linestyle='--', alpha=0.5,
                label=f'Baseline Accuracy: {data["baseline_accuracy"]:.4f}')
    
    ax5.set_title('Accuracy Over Steps (All Runs)')
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Accuracy')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.grid(True, alpha=0.3)
    
    # Adjust layout and add metadata
    plt.suptitle(f'Hill Climbing Analysis - {data["model_name"]}', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('hillclimbing_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_summary_statistics(data):
    """Print summary statistics for the hill climbing runs"""
    print(f"Model: {data['model_name']}")
    print(f"Baseline Accuracy: {data['baseline_accuracy']:.4f}")
    print("\nRun Statistics:")
    
    for run_idx, run in enumerate(data['runs']):
        print(f"\nRun {run_idx + 1}:")
        initial_acc = run['initial_accuracy']
        final_acc = run['steps'][-1]['accuracy'] if run['steps'] else None
        initial_layers = sum(run['initial_state'])
        final_layers = sum(run['steps'][-1]['state']) if run['steps'] else None
        
        print(f"- Initial Accuracy: {initial_acc:.4f}")
        print(f"- Final Accuracy: {final_acc:.4f}")
        print(f"- Initial Active Layers: {initial_layers}")
        print(f"- Final Active Layers: {final_layers}")
        print(f"- Number of Steps: {len(run['steps'])}")

def main():
    # Load and visualize the data
    data = load_data('hermes-llama-3.1-hc.json')
    create_visualizations(data)
    print_summary_statistics(data)

if __name__ == "__main__":
    main()

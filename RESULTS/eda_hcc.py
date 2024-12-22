import matplotlib.pyplot as plt
import json
import numpy as np

def load_data(filename):
    """Load and parse JSON data"""
    with open(filename, 'r') as f:
        return json.load(f)

def create_visualization(data):
    
    for run_idx, run in enumerate(data['runs']):
        steps = [step['step'] for step in run['steps']]
        accuracies = [step['accuracy'] for step in run['steps']]
        
        plt.plot(steps, accuracies, marker='o', linestyle='-', linewidth=2, 
                markersize=6, alpha=0.7,
                label=f'Run {run_idx + 1}')
        plt.show()

def main():
    # Load and visualize the data
    data = load_data('hermes-llama-3.1-hc.json')
    create_visualization(data)

if __name__ == "__main__":
    main()

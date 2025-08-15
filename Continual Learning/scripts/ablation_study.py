"""
Ablation study to test the localization hypothesis:
- Attention layers adapt to "what" (objects)
- FFN layers adapt to "how" (representation)
"""

import os
import sys
import torch
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_cifar100 import main as train_main


def run_ablation_study():
    """
    Run complete ablation study comparing:
    1. Attention-only LoRA
    2. FFN-only LoRA
    3. Both (attention + FFN)
    """
    
    # Base configuration
    base_args = {
        'model_name': 'vit_base_patch16_224',
        'lora_rank': 4,
        'lora_alpha': 4.0,
        'data_root': './data',
        'num_tasks': 10,
        'classes_per_task': 10,
        'batch_size': 128,
        'num_epochs': 20,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'lambda_unknown': 0.5,
        'buffer_size': 2000,
        'samples_per_class': 20,
        'selection_strategy': 'herding',
        'device': 'cuda',
        'seed': 42,
        'save_dir': './results/ablation'
    }
    
    # Configurations to test
    configs = [
        ('attention_only', 'Attention-Only (What)'),
        ('ffn_only', 'FFN-Only (How)'),
        ('both', 'Both (What + How)')
    ]
    
    results_summary = {}
    
    print("\n" + "="*60)
    print("ABLATION STUDY: Testing Localization Hypothesis")
    print("="*60 + "\n")
    
    for config_name, config_desc in configs:
        print(f"\n{'='*50}")
        print(f"Testing: {config_desc}")
        print(f"{'='*50}\n")
        
        # Create args namespace
        args = argparse.Namespace(**base_args)
        args.lora_config = config_name
        args.experiment_name = f"ablation_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Run training with this configuration
        sys.argv = ['train_cifar100.py']  # Reset argv
        for key, value in vars(args).items():
            if key != 'experiment_name':
                sys.argv.extend([f'--{key}', str(value)])
        sys.argv.extend(['--experiment_name', args.experiment_name])
        
        # Import and run main training
        import subprocess
        result = subprocess.run(
            [sys.executable, 'scripts/train_cifar100.py'] + sys.argv[1:],
            capture_output=True,
            text=True
        )
        
        # Load results
        results_path = os.path.join(args.save_dir, args.experiment_name, 'results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
                results_summary[config_name] = {
                    'description': config_desc,
                    'average_accuracy': results['average_accuracy'],
                    'average_forgetting': results['average_forgetting'],
                    'final_accuracies': results['final_accuracies']
                }
    
    # Compare and visualize results
    compare_results(results_summary, base_args['save_dir'])


def compare_results(results_summary, save_dir):
    """Compare and visualize ablation results"""
    
    if not results_summary:
        print("No results to compare!")
        return
    
    # Create comparison plots
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Average accuracy comparison
    ax = axes[0]
    configs = list(results_summary.keys())
    accuracies = [results_summary[c]['average_accuracy'] for c in configs]
    labels = [results_summary[c]['description'] for c in configs]
    
    bars = ax.bar(range(len(configs)), accuracies, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax.set_ylabel('Average Accuracy (%)')
    ax.set_title('Average Accuracy Comparison')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # 2. Average forgetting comparison
    ax = axes[1]
    forgetting = [results_summary[c]['average_forgetting'] for c in configs]
    
    bars = ax.bar(range(len(configs)), forgetting, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax.set_ylabel('Average Forgetting (%)')
    ax.set_title('Average Forgetting Comparison')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Add value labels
    for bar, forg in zip(bars, forgetting):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{forg:.1f}%', ha='center', va='bottom')
    
    # 3. Per-task accuracy comparison
    ax = axes[2]
    
    # Prepare data for grouped bar plot
    num_tasks = 10
    x = np.arange(num_tasks)
    width = 0.25
    
    for i, config in enumerate(configs):
        task_accs = [results_summary[config]['final_accuracies'][f'task_{j}'] 
                     for j in range(num_tasks)]
        offset = (i - 1) * width
        ax.bar(x + offset, task_accs, width, 
               label=results_summary[config]['description'][:10],
               alpha=0.8)
    
    ax.set_xlabel('Task ID')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Task Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'T{i}' for i in range(num_tasks)])
    ax.legend(loc='best', fontsize=8)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    comparison_path = os.path.join(save_dir, 'ablation_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary table
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("="*60)
    print(f"{'Configuration':<25} {'Avg Acc (%)':<15} {'Avg Forget (%)':<15}")
    print("-"*55)
    
    for config in configs:
        desc = results_summary[config]['description']
        acc = results_summary[config]['average_accuracy']
        forg = results_summary[config]['average_forgetting']
        print(f"{desc:<25} {acc:<15.2f} {forg:<15.2f}")
    
    print("\n" + "="*60)
    
    # Save summary
    summary_path = os.path.join(save_dir, 'ablation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print(f"\nResults saved to:")
    print(f"  - Plot: {comparison_path}")
    print(f"  - Summary: {summary_path}")
    
    # Analyze hypothesis
    analyze_hypothesis(results_summary)


def analyze_hypothesis(results_summary):
    """Analyze whether results support the localization hypothesis"""
    
    print("\n" + "="*60)
    print("HYPOTHESIS ANALYSIS")
    print("="*60)
    
    if 'attention_only' in results_summary and 'ffn_only' in results_summary:
        att_acc = results_summary['attention_only']['average_accuracy']
        ffn_acc = results_summary['ffn_only']['average_accuracy']
        att_forg = results_summary['attention_only']['average_forgetting']
        ffn_forg = results_summary['ffn_only']['average_forgetting']
        
        print("\nKey Findings:")
        print(f"1. Attention-only accuracy: {att_acc:.2f}%")
        print(f"   FFN-only accuracy: {ffn_acc:.2f}%")
        print(f"   Difference: {att_acc - ffn_acc:+.2f}%")
        
        print(f"\n2. Attention-only forgetting: {att_forg:.2f}%")
        print(f"   FFN-only forgetting: {ffn_forg:.2f}%")
        print(f"   Difference: {att_forg - ffn_forg:+.2f}%")
        
        # Interpretation
        print("\nInterpretation:")
        
        if att_acc > ffn_acc and att_forg < ffn_forg:
            print("✓ HYPOTHESIS SUPPORTED:")
            print("  Attention-only adaptation shows better accuracy with less forgetting,")
            print("  suggesting attention layers are better suited for task-specific")
            print("  'what' adaptation while maintaining stability.")
            
        elif ffn_acc > att_acc:
            print("✗ HYPOTHESIS NOT SUPPORTED:")
            print("  FFN-only adaptation shows better performance, suggesting")
            print("  representation learning ('how') might be more important")
            print("  for this task than object-level adaptation.")
            
        else:
            print("~ MIXED RESULTS:")
            print("  The results show trade-offs between accuracy and forgetting.")
            print("  Further investigation needed to understand the mechanisms.")
        
        if 'both' in results_summary:
            both_acc = results_summary['both']['average_accuracy']
            both_forg = results_summary['both']['average_forgetting']
            
            print(f"\n3. Combined (both) performance:")
            print(f"   Accuracy: {both_acc:.2f}%")
            print(f"   Forgetting: {both_forg:.2f}%")
            
            if both_acc > max(att_acc, ffn_acc):
                print("   ✓ Synergistic effect: Combined adaptation outperforms individual")
            else:
                print("   ✗ No synergy: Combined adaptation doesn't improve performance")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    run_ablation_study()
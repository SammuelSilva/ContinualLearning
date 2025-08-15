"""
Main training script for CIFAR-100 continual learning with LoRA-ViT.
Tests different hypotheses about attention vs FFN adaptation.
"""

import os
import sys
import torch
import argparse
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.lora_vit import ContinualLoRAViT
from src.data.cifar_dataset import ContinualCIFAR100
from src.data.memory_buffer import MemoryBuffer
from src.trainers.continual_trainer import ContinualTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train LoRA-ViT on CIFAR-100')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224',
                       help='ViT model name from timm')
    parser.add_argument('--lora_rank', type=int, default=4,
                       help='Rank for LoRA decomposition')
    parser.add_argument('--lora_alpha', type=float, default=4.0,
                       help='LoRA scaling factor')
    parser.add_argument('--lora_config', type=str, default='attention_only',
                       choices=['attention_only', 'ffn_only', 'both'],
                       help='LoRA placement configuration (hypothesis testing)')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='./data',
                       help='Root directory for datasets')
    parser.add_argument('--num_tasks', type=int, default=10,
                       help='Number of tasks to split CIFAR-100')
    parser.add_argument('--classes_per_task', type=int, default=10,
                       help='Number of classes per task')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of epochs per task')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--lambda_unknown', type=float, default=0.5,
                       help='Weight for unknown class loss')
    
    # Memory buffer arguments
    parser.add_argument('--buffer_size', type=int, default=2000,
                       help='Total memory buffer size')
    parser.add_argument('--samples_per_class', type=int, default=20,
                       help='Samples per class in buffer')
    parser.add_argument('--selection_strategy', type=str, default='herding',
                       choices=['random', 'herding'],
                       help='Buffer selection strategy')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for this experiment')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def calculate_forgetting(task_accuracies, current_task):
    """Calculate average forgetting metric"""
    forgetting = []
    
    for task_id in range(current_task):
        task_key = f"task_{task_id}"
        # Best accuracy when task was learned
        best_acc = task_accuracies[task_id][task_key]
        # Current accuracy
        current_acc = task_accuracies[current_task][task_key]
        # Forgetting for this task
        task_forgetting = best_acc - current_acc
        forgetting.append(max(0, task_forgetting))  # Forgetting can't be negative
    
    return np.mean(forgetting) if forgetting else 0


def plot_results(task_accuracies, final_results, experiment_dir, lora_config):
    """Plot training results"""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 5)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Task accuracy evolution
    ax = axes[0]
    num_tasks = len(task_accuracies)
    
    # Create matrix for heatmap
    accuracy_matrix = np.zeros((num_tasks, num_tasks))
    for train_task in range(num_tasks):
        for eval_task in range(min(train_task + 1, num_tasks)):
            task_key = f"task_{eval_task}"
            if task_key in task_accuracies[train_task]:
                accuracy_matrix[train_task, eval_task] = task_accuracies[train_task][task_key]
    
    # Mask upper triangle
    mask = np.triu(np.ones_like(accuracy_matrix), k=1)
    
    sns.heatmap(
        accuracy_matrix,
        mask=mask,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        vmin=0,
        vmax=100,
        ax=ax,
        cbar_kws={'label': 'Accuracy (%)'}
    )
    ax.set_xlabel('Task ID')
    ax.set_ylabel('Training Progress')
    ax.set_title(f'Task Accuracy Evolution\n({lora_config})')
    
    # 2. Final accuracies per task
    ax = axes[1]
    tasks = list(final_results.keys())
    accuracies = list(final_results.values())
    
    bars = ax.bar(range(len(tasks)), accuracies, color='steelblue', alpha=0.8)
    ax.set_xlabel('Task ID')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Final Task Accuracies\n(Avg: {np.mean(accuracies):.1f}%)')
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([f"T{i}" for i in range(len(tasks))])
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Forgetting analysis
    ax = axes[2]
    
    # Calculate per-task forgetting
    forgetting_per_task = []
    for task_id in range(num_tasks - 1):  # No forgetting for last task
        task_key = f"task_{task_id}"
        best_acc = task_accuracies[task_id][task_key]
        final_acc = final_results[task_key]
        forgetting = max(0, best_acc - final_acc)
        forgetting_per_task.append(forgetting)
    
    if forgetting_per_task:
        bars = ax.bar(range(len(forgetting_per_task)), forgetting_per_task, 
                      color='coral', alpha=0.8)
        ax.set_xlabel('Task ID')
        ax.set_ylabel('Forgetting (%)')
        ax.set_title(f'Task Forgetting\n(Avg: {np.mean(forgetting_per_task):.1f}%)')
        ax.set_xticks(range(len(forgetting_per_task)))
        ax.set_xticklabels([f"T{i}" for i in range(len(forgetting_per_task))])
        
        # Add value labels
        for bar, forget in zip(bars, forgetting_per_task):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{forget:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, 'results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to: {os.path.join(experiment_dir, 'results.png')}")


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create experiment name
    if args.experiment_name is None:
        args.experiment_name = f"cifar100_{args.lora_config}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create directories
    experiment_dir = os.path.join(args.save_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    print(f"\n{'='*50}")
    print(f"Experiment: {args.experiment_name}")
    print(f"LoRA Configuration: {args.lora_config}")
    print(f"{'='*50}\n")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print("Setting up CIFAR-100 dataset...")
    dataset = ContinualCIFAR100(
        data_root=args.data_root,
        num_tasks=args.num_tasks,
        classes_per_task=args.classes_per_task,
        seed=args.seed
    )
    
    # Create model
    print("Creating LoRA-ViT model...")
    model = ContinualLoRAViT(
        vit_model_name=args.model_name,
        num_classes_per_task=None,  # Will add tasks dynamically
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_config=args.lora_config,
        use_pretrained=True
    )
    
    # Create memory buffer
    print("Initializing memory buffer...")
    memory_buffer = MemoryBuffer(
        buffer_size=args.buffer_size,
        selection_strategy=args.selection_strategy,
        samples_per_class=args.samples_per_class
    )
    
    # Create trainer
    trainer = ContinualTrainer(
        model=model,
        memory_buffer=memory_buffer,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lambda_unknown=args.lambda_unknown,
        save_dir=checkpoint_dir
    )
    
    # Results storage
    task_accuracies = {}
    forgetting_metrics = []
    
    # Train on each task sequentially
    for task_idx in range(args.num_tasks):
        task_id = f"task_{task_idx}"
        
        print(f"\n{'='*50}")
        print(f"TASK {task_idx + 1}/{args.num_tasks}")
        print(f"{'='*50}")
        
        # Add new task to model
        model.add_task(task_id, args.classes_per_task)
        
        # Get data loaders for current task
        train_loader, val_loader, test_loader = dataset.get_task_loaders(
            task_idx,
            batch_size=args.batch_size
        )
        
        # Train on current task
        best_acc = trainer.train_task(
            task_id=task_id,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs
        )
        
        # Evaluate on all tasks seen so far
        print(f"\nEvaluating on all {task_idx + 1} tasks...")
        test_loaders = {}
        for i in range(task_idx + 1):
            _, _, test_loader_i = dataset.get_task_loaders(i, batch_size=args.batch_size)
            test_loaders[f"task_{i}"] = test_loader_i
        
        current_results = trainer.evaluate_all_tasks(test_loaders)
        task_accuracies[task_idx] = current_results
        
        # Calculate forgetting
        if task_idx > 0:
            forgetting = calculate_forgetting(task_accuracies, task_idx)
            forgetting_metrics.append(forgetting)
            print(f"Average Forgetting: {forgetting:.2f}%")
    
    # Final evaluation on all tasks
    print(f"\n{'='*50}")
    print("FINAL EVALUATION ON ALL TASKS")
    print(f"{'='*50}")
    
    all_test_loaders = {}
    for i in range(args.num_tasks):
        _, _, test_loader = dataset.get_task_loaders(i, batch_size=args.batch_size)
        all_test_loaders[f"task_{i}"] = test_loader
    
    final_results = trainer.evaluate_all_tasks(all_test_loaders)
    
    # Save results
    results = {
        'args': vars(args),
        'task_accuracies': task_accuracies,
        'final_accuracies': final_results,
        'forgetting_metrics': forgetting_metrics,
        'average_accuracy': np.mean(list(final_results.values())),
        'average_forgetting': np.mean(forgetting_metrics) if forgetting_metrics else 0
    }
    
    with open(os.path.join(experiment_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Plot results
    plot_results(task_accuracies, final_results, experiment_dir, args.lora_config)
    
    # Print summary
    print(f"\n{'='*50}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*50}")
    print(f"LoRA Configuration: {args.lora_config}")
    print(f"Average Accuracy: {results['average_accuracy']:.2f}%")
    print(f"Average Forgetting: {results['average_forgetting']:.2f}%")
    print(f"Results saved to: {experiment_dir}")
    
    # Print per-task results
    print(f"\n{'='*50}")
    print("PER-TASK FINAL ACCURACIES")
    print(f"{'='*50}")
    for task_id, acc in final_results.items():
        print(f"{task_id}: {acc:.2f}%")
    
    print(f"\n{'='*50}")
    print("Training completed successfully!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
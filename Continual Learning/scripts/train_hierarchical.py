"""
Training script for Hierarchical LoRA-ViT with TRIM merging and Unknown Data Handling.
Supports both standard and hierarchical modes with comprehensive experiments.
"""

import os
import sys
import torch
import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.hierarchical_lora import HierarchicalLoRAViT
from src.models.lora_vit import ContinualLoRAViT
from src.data.cifar_dataset import EnhancedContinualCIFAR100
from src.data.memory_buffer import MemoryBuffer
from src.trainers.hierarchical_trainer import HierarchicalTrainer
from src.trainers.continual_trainer import ContinualTrainer
from src.utils.visualization import HierarchicalVisualizer
from src.utils.helpers import NumpyJSONEncoder

def setup_logging(experiment_dir: str):
    """Setup logging configuration"""
    log_file = os.path.join(experiment_dir, 'training.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Hierarchical LoRA-ViT with TRIM merging and Unknown Data'
    )
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224',
                       choices=['vit_tiny_patch16_224', 'vit_small_patch16_224', 
                               'vit_base_patch16_224', 'vit_large_patch16_224'],
                       help='ViT model name from timm')
    parser.add_argument('--lora_rank', type=int, default=4,
                       help='Rank for LoRA decomposition')
    parser.add_argument('--lora_alpha', type=float, default=4.0,
                       help='LoRA scaling factor')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout rate')
    parser.add_argument('--lora_config', type=str, default='attention_only',
                       choices=['attention_only', 'ffn_only', 'both'],
                       help='LoRA placement configuration')
    parser.add_argument('--cpu_validation', action='store_true', default=True,
                        help='Run validation on CPU to save GPU memory')
    parser.add_argument('--cpu_buffer_update', action='store_true', default=True,
                        help='Update memory buffer using CPU features')
    # Hierarchical arguments
    parser.add_argument('--use_hierarchical', action='store_true',
                       help='Use hierarchical architecture with block merging')
    parser.add_argument('--max_tasks_per_block', type=int, default=2,
                       help='Maximum tasks allowed per block')
    parser.add_argument('--min_tasks_to_merge', type=int, default=2,
                       help='Minimum tasks needed to attempt merge')
    parser.add_argument('--similarity_threshold', type=float, default=0.8,
                       help='Cosine similarity threshold for grouping')
    parser.add_argument('--trim_percentage', type=float, default=0.1,
                       help='Percentage of weights to keep in TIES')
    parser.add_argument('--max_accuracy_drop', type=float, default=2.0,
                       help='Maximum accuracy drop allowed')
    parser.add_argument('--max_rejection_drop', type=float, default=4.0,
                       help='Maximum rejection rate drop allowed')
    parser.add_argument('--max_fp_increase', type=float, default=2.0,
                       help='Maximum false positive increase allowed')
    parser.add_argument('--ablation_samples', type=int, default=1000,
                       help='Number of samples for ablation search')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='./data',
                       help='Root directory for datasets')
    parser.add_argument('--dataset', type=str, default='cifar100',
                       choices=['cifar100', 'imagenet-r', 'core50'],
                       help='Dataset to use')
    parser.add_argument('--num_tasks', type=int, default=10,
                       help='Number of tasks to split dataset')
    parser.add_argument('--classes_per_task', type=int, default=10,
                       help='Number of classes per task')
    parser.add_argument('--validation_split', type=float, default=0.1,
                       help='Validation split ratio')
    
    # Unknown data arguments
    parser.add_argument('--use_unknown_data', action='store_true',
                       help='Use SVHN as unknown/OOD data for training')
    parser.add_argument('--unknown_ratio', type=float, default=0.3,
                       help='Ratio of unknown samples to task samples (for task 0)')
    parser.add_argument('--unknown_ratio_decay', type=float, default=0.85,
                       help='Decay factor for unknown ratio across tasks')
    parser.add_argument('--include_unknown_test', action='store_true',
                       help='Include unknown samples in test evaluation')
    parser.add_argument('--unknown_temperature', type=float, default=2.0,
                       help='Temperature scaling for unknown detection')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of epochs per task')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                       help='Warmup epochs for new blocks')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                       help='Gradient clipping value')
    
    # Continual learning arguments
    parser.add_argument('--lambda_task_unknown', type=float, default=0.5,
                       help='Weight for task-level unknown class loss')
    parser.add_argument('--lambda_block_unknown', type=float, default=0.3,
                       help='Weight for block-level unknown class loss')
    parser.add_argument('--lambda_classification', type=float, default=1.0,
                       help='Weight for classification loss')
    parser.add_argument('--lambda_intra_block', type=float, default=0.1,
                       help='Weight for intra-block regularization')
    
    # Memory buffer arguments
    parser.add_argument('--buffer_size', type=int, default=2000,
                       help='Total memory buffer size')
    parser.add_argument('--samples_per_class', type=int, default=20,
                       help='Samples per class in buffer')
    parser.add_argument('--selection_strategy', type=str, default='herding',
                       choices=['random', 'herding', 'uncertainty'],
                       help='Buffer selection strategy')
    
    # Experiment arguments
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for this experiment')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--checkpoint_freq', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')
    
    # Resume arguments
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    parser.add_argument('--resume_dir', type=str, default=None,
                       help='Directory to resume from')
    
    # Evaluation arguments
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate, no training')
    parser.add_argument('--eval_checkpoint', type=str, default=None,
                       help='Checkpoint path for evaluation')
    
    # Visualization arguments
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations during training')
    parser.add_argument('--plot_freq', type=int, default=1,
                       help='Create plots every N tasks')
    parser.add_argument('--create_animation', action='store_true',
                       help='Create accuracy evolution animation')
    
    # Ablation study arguments
    parser.add_argument('--ablation_mode', type=str, default=None,
                       choices=['no_orthogonal', 'no_unknown', 'no_hierarchy', 
                               'different_ranks', 'different_blocks', 'unknown_ratios'],
                       help='Run specific ablation study')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(args) -> torch.nn.Module:
    """Create model based on arguments"""
    
    if args.use_hierarchical:
        print("🚀 Creating Hierarchical LoRA-ViT with Intelligent TIES Merging")
        
        from src.models.merge_strategies import MergeConfig
        merge_config = MergeConfig(
            similarity_threshold=args.similarity_threshold,
            trim_percentage=args.trim_percentage,
            max_accuracy_drop=args.max_accuracy_drop,
            max_rejection_drop=args.max_rejection_drop,
            max_fp_increase=args.max_fp_increase,
            ablation_samples=args.ablation_samples
        )

        model = HierarchicalLoRAViT(
            vit_model_name=args.model_name,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_config=args.lora_config,
            use_pretrained=True,
            max_tasks_per_block=args.max_tasks_per_block,
            min_tasks_to_merge=args.min_tasks_to_merge,
            merge_config=merge_config
        )

        print(f"  - Tasks per block: {args.max_tasks_per_block}")
        if args.use_unknown_data:
            print(f"  - Unknown detection: ENABLED")
    else:
        print("Error: Hierarchical LoRA is not enabled")
        return None
    
    print(f"  - Model: {args.model_name}")
    print(f"  - LoRA rank: {args.lora_rank}")
    print(f"  - LoRA config: {args.lora_config}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    return model


def create_dataset(args):
    """Create dataset based on arguments"""
    
    if args.dataset == 'cifar100':
        # Always use Enhanced dataset which can handle both cases
        dataset = EnhancedContinualCIFAR100(
            data_root=args.data_root,
            num_tasks=args.num_tasks,
            classes_per_task=args.classes_per_task,
            validation_split=args.validation_split,
            unknown_ratio=args.unknown_ratio,
            unknown_ratio_decay=args.unknown_ratio_decay,
            use_unknown_data=args.use_unknown_data,
            seed=args.seed,
            cache_current_task_only=True
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")
    
    print(f"📊 Dataset: {args.dataset}")
    print(f"  - Num tasks: {args.num_tasks}")
    print(f"  - Classes per task: {args.classes_per_task}")
    
    if args.use_unknown_data:
        print(f"  - Unknown data: SVHN")
        print(f"  - Unknown ratio (task 0): {args.unknown_ratio:.1%}")
        print(f"  - Unknown decay: {args.unknown_ratio_decay}")
    
    return dataset


def create_trainer(model, memory_buffer, args):
    """Create appropriate trainer based on model type and configuration"""
    
    if args.use_hierarchical:
        # Use enhanced trainer when unknown data is enabled
        if args.use_unknown_data:
            trainer = HierarchicalTrainer(
                model=model,
                memory_buffer=memory_buffer,
                device=torch.device(args.device),
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                lambda_task_unknown=args.lambda_task_unknown,
                lambda_block_unknown=args.lambda_block_unknown,
                lambda_classification=args.lambda_classification,
                unknown_temperature=args.unknown_temperature,
                save_dir=os.path.join(args.experiment_dir, 'checkpoints'),
                num_tasks=args.num_tasks
            )
    else:
        trainer = ContinualTrainer(
            model=model,
            memory_buffer=memory_buffer,
            device=torch.device(args.device),
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lambda_unknown=args.lambda_task_unknown,
            save_dir=os.path.join(args.experiment_dir, 'checkpoints'),
            num_tasks=args.num_tasks
        )
    
    return trainer

def run_training(args, model, dataset, trainer, memory_buffer, logger):
    """
    Main training loop with unknown data handling and buffer management
    Modified for memory efficiency - clears cache between tasks
    """
    
        # Visualization setup
    visualizer = None
    if args.visualize:
        visualizer = HierarchicalVisualizer(save_dir=args.experiment_dir)
    
    # Track metrics
    all_results = {}
    task_accuracies = {}
    unknown_metrics = {}
    
    # Resume state
    start_task = 0
    if args.resume:
        checkpoint_state = load_checkpoint_state(args.resume_dir or args.experiment_dir)
        if checkpoint_state:
            start_task = checkpoint_state['last_completed_task'] + 1
            task_accuracies = checkpoint_state.get('task_accuracies', {})
            unknown_metrics = checkpoint_state.get('unknown_metrics', {})
            print(f"Resuming from task {start_task}")
    
    # Training loop
    for task_idx in range(start_task, args.num_tasks):
        task_id = f"task_{task_idx}"
        
        print(f"\n{'='*60}")
        print(f"TASK {task_idx + 1}/{args.num_tasks}: {task_id}")
        print(f"{'='*60}")
        
        # MEMORY MANAGEMENT: Clear everything before starting new task
        dataset.clear_cache()  # Clear dataset cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Get data loaders with unknown data if enabled
        # Use fewer workers and no pin_memory for efficiency
        train_loader, val_loader, test_loader = dataset.get_task_loaders(
            task_idx,
            batch_size=args.batch_size,
            num_workers=min(args.num_workers, 2),  # Limit workers for memory
            include_unknown_train=args.use_unknown_data,
            include_unknown_test=args.include_unknown_test,
            pin_memory=False  # IMPORTANT: Keep data on CPU
        )
        
        # Log dataset statistics
        if args.use_unknown_data:
            stats = dataset.get_statistics(task_idx)
            print(f"Task {task_idx} dataset composition:")
            print(f"  - Unknown ratio: {stats['unknown_ratio']:.2%}")
            print(f"  - Task samples: {stats.get('task_samples', 'N/A')}")
            print(f"  - Expected unknown samples: {stats.get('expected_unknown_samples', 'N/A')}")
        
        # Add task with correct number of classes
        num_classes = args.classes_per_task + 1 if args.use_unknown_data else args.classes_per_task
        model.add_task(task_id, num_classes)
        
        # Set memory buffer reference in model if hierarchical
        if args.use_hierarchical and hasattr(model, 'set_memory_buffer'):
            model.set_memory_buffer(memory_buffer)
        
        # Train on current task
        print(f"Training on {task_id}...")
        best_acc = trainer.train_task(
            task_id=task_id,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            patience=5,
            task_idx=task_idx,
            warmup_epochs=args.warmup_epochs if task_idx == 0 else 0
        )
        
        print(f"Best validation accuracy: {best_acc:.2f}%")
        
        # Mark task as trained (for hierarchical model)
        if args.use_hierarchical:
            model.mark_task_trained(task_id)
        
        # UPDATE MEMORY BUFFER with current task data
        if memory_buffer is not None:
            update_memory_buffer_efficient(
                model, 
                memory_buffer, 
                train_loader, 
                task_id, 
                args,
                num_batches=3  # Reduced from 5 to 3 for memory
            )
        
        # MEMORY MANAGEMENT: Clear cache after buffer update
        dataset.clear_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                    
        # EVALUATE ON BUFFER after updating (skip first task)
        if memory_buffer is not None and task_idx > 0 and len(memory_buffer) > 0:
            print(f"\n🔍 Evaluating model on memory buffer...")
            buffer_metrics = trainer.evaluate_buffer_metrics(task_id, task_idx)
            
            if not hasattr(trainer, 'buffer_metrics_history'):
                trainer.buffer_metrics_history = {}
            trainer.buffer_metrics_history[task_idx] = buffer_metrics
        
        # Store unknown metrics if available
        if args.use_unknown_data and hasattr(trainer, 'get_unknown_metrics'):
            unknown_metrics[task_idx] = trainer.get_unknown_metrics(task_id)
            print(f"Unknown detection metrics:")
            for metric_name, value in unknown_metrics[task_idx].items():
                print(f"  - {metric_name}: {value:.3f}")
        
        # Evaluate on all tasks (one at a time to save memory)
        print(f"\nEvaluating on all {task_idx + 1} tasks...")
        current_results = {}
        
        for i in range(task_idx + 1):
            # MEMORY MANAGEMENT: Clear cache before loading next task
            dataset.clear_cache()
            
            # Get loader for single task
            _, _, test_loader_i = dataset.get_task_loaders(
                i,
                batch_size=args.batch_size,
                num_workers=min(args.num_workers, 2),
                include_unknown_train=False,
                include_unknown_test=args.include_unknown_test,
                pin_memory=False  # Keep on CPU
            )
            task_i_id = f"task_{i}"
            
            # Evaluate immediately and store result
            with torch.no_grad():
                acc = trainer.evaluate_all_tasks(test_loader_i, task_i_id)
                current_results[task_i_id] = acc
            
            # MEMORY MANAGEMENT: Clear loader from memory
            del test_loader_i
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        task_accuracies[task_idx] = current_results
        
        # MEMORY MANAGEMENT: Clear dataset cache after evaluation
        dataset.clear_cache()
        
        # Save checkpoint state
        save_checkpoint_state(
            args.experiment_dir,
            task_idx,
            task_accuracies,
            trainer.metrics,
            unknown_metrics
        )
        
        # Visualizations
        if visualizer and task_idx % args.plot_freq == 0:
            create_visualizations(
                visualizer, model, trainer, 
                task_idx, args, unknown_metrics
            )
        
        # Log statistics
        if args.use_hierarchical:
            stats = model.get_statistics()
            print(f"\nHierarchical Statistics:")
            print(f"  - Merged blocks: {stats['num_merged_blocks']}")
            print(f"  - Specialist tasks (unmerged): {stats['specialist_tasks']}")
            print(f"  - Total tasks: {stats['total_tasks']}")
            print(f"  - Merge attempts: {stats['merge_attempts']}")
            print(f"  - Successful merges: {stats['successful_merges']}")
        
        # MEMORY MANAGEMENT: Final cleanup for this task
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final evaluation
    print(f"\n{'='*60}")
    print("FINAL EVALUATION ON ALL TASKS")
    print(f"{'='*60}")
    
    # MEMORY MANAGEMENT: Clear cache before final evaluation
    dataset.clear_cache()
    
    # Evaluate one task at a time for memory efficiency
    final_results = {}
    for i in range(args.num_tasks):
        # MEMORY MANAGEMENT: Clear cache before each task
        dataset.clear_cache()
        
        _, _, test_loader = dataset.get_task_loaders(
            i,
            batch_size=args.batch_size,
            num_workers=min(args.num_workers, 2),
            include_unknown_train=False,
            include_unknown_test=args.include_unknown_test,
            pin_memory=False  # Keep on CPU
        )
        task_i_id = f"task_{i}"
        
        with torch.no_grad():
            if args.use_hierarchical and hasattr(trainer, 'evaluate_hierarchical'):
                acc = trainer.evaluate_hierarchical({task_i_id: test_loader})[task_i_id]
            else:
                acc = trainer.evaluate_all_tasks(test_loader, task_i_id)
            final_results[task_i_id] = acc
        
        # MEMORY MANAGEMENT: Clear after each evaluation
        del test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # MEMORY MANAGEMENT: Unload all datasets after training completes
    dataset.unload_all_datasets()
    
    # Get comprehensive metrics
    final_metrics = trainer.get_metrics_summary()
    
    # Save results
    results = {
        'args': vars(args),
        'task_accuracies': task_accuracies,
        'final_results': final_results,
        'metrics': final_metrics,
        'unknown_metrics': unknown_metrics if args.use_unknown_data else None,
        'model_stats': model.get_statistics() if args.use_hierarchical else None
    }
    
    results_path = os.path.join(args.experiment_dir, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4, cls=NumpyJSONEncoder)
    
    # Print summary
    print_summary(final_metrics, final_results, args, unknown_metrics)
    
    # Final visualizations
    if visualizer:
        create_final_visualizations(
            visualizer, model, trainer,
            args, unknown_metrics
        )
    
    # MEMORY MANAGEMENT: Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results

def create_visualizations(visualizer, model, trainer, task_idx, args, unknown_metrics=None):
    """Create visualizations during training"""
    
    save_dir = os.path.join(args.experiment_dir, 'visualizations', f'task_{task_idx}')
    os.makedirs(save_dir, exist_ok=True)
    
    if args.use_hierarchical:
        # Plot hierarchy tree
        stats = model.get_statistics()
        visualizer.plot_hierarchy_tree(
            stats,
            save_path=os.path.join(save_dir, 'hierarchy.html')
        )
        
        # Plot memory efficiency
        visualizer.plot_memory_efficiency(
            num_tasks=task_idx + 1,
            tasks_per_block=args.max_tasks_per_block,
            lora_rank=args.lora_rank,
            save_path=os.path.join(save_dir, 'memory_efficiency.png')
        )
        
        # Plot orthogonality matrix
        if len(model.task_adapters) > 1:
            visualizer.plot_orthogonality_matrix(
                model.task_adapters,
                save_path=os.path.join(save_dir, 'orthogonality.png')
            )
    
    # Plot unknown detection metrics if available
    if args.use_unknown_data and unknown_metrics:
        visualizer.plot_unknown_metrics(
            unknown_metrics,
            save_path=os.path.join(save_dir, 'unknown_detection.png')
        )
    
    # Plot metrics evolution
    trainer.metrics.plot_metrics_evolution(
        save_path=os.path.join(save_dir, f'metrics_evolution_({task_idx+1}).png')
    )


def create_final_visualizations(visualizer, model, trainer, args, unknown_metrics=None):
    """Create final visualizations after training"""
    
    save_dir = os.path.join(args.experiment_dir, 'visualizations', 'final')
    os.makedirs(save_dir, exist_ok=True)
    
    # Create training dashboard
    if args.use_hierarchical:
        stats = model.get_statistics()
        visualizer.create_training_dashboard(
            metrics_history=trainer.metrics.__dict__,
            hierarchy_stats=stats,
            unknown_metrics=unknown_metrics,
            save_path=os.path.join(save_dir, 'dashboard.html')
        )
    
    # Create accuracy animation
    if args.create_animation:
        from src.utils.visualization import MetricsAnimator
        animator = MetricsAnimator()
        animator.create_accuracy_animation(
            trainer.metrics.accuracy_matrix,
            save_path=os.path.join(save_dir, 'accuracy_evolution.gif')
        )
    
    # Save final plots
    trainer.metrics.plot_accuracy_matrix(
        save_path=os.path.join(save_dir, 'accuracy_matrix.png')
    )
    trainer.metrics.plot_metrics_evolution(
        save_path=os.path.join(save_dir, 'metrics_evolution.png')
    )
    
    # Plot final unknown metrics
    if args.use_unknown_data and unknown_metrics:
        visualizer.plot_unknown_metrics_summary(
            unknown_metrics,
            save_path=os.path.join(save_dir, 'unknown_summary.png')
        )


def print_summary(metrics, final_results, args, unknown_metrics=None):
    """Print training summary with unknown metrics"""
    
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    if args.use_hierarchical:
        print(f"Architecture: Hierarchical LoRA-ViT")
        print(f"  - Tasks per block: {args.max_tasks_per_block}")
        if args.use_unknown_data:
            print(f"  - Unknown detection: ENABLED")
    else:
        print(f"Architecture: Standard Continual LoRA-ViT")
    
    print(f"\nLoRA Configuration: {args.lora_config}")
    print(f"  - Rank: {args.lora_rank}")
    print(f"  - Alpha: {args.lora_alpha}")
    
    if args.use_unknown_data:
        print(f"\nUnknown Data Configuration:")
        print(f"  - Dataset: SVHN")
        print(f"  - Initial ratio: {args.unknown_ratio:.1%}")
        print(f"  - Decay factor: {args.unknown_ratio_decay}")
    
    print(f"\n{'='*60}")
    print("METRICS")
    print(f"{'='*60}")
    print(f"Average Accuracy:     {metrics['average_accuracy']:.2f}%")
    print(f"Average Forgetting:   {metrics['average_forgetting']:.2f}%")
    print(f"Backward Transfer:    {metrics['backward_transfer']:.2f}%")
    print(f"Forward Transfer:     {metrics['forward_transfer']:.2f}%")
    print(f"Plasticity:          {metrics['plasticity']:.2f}%")
    print(f"Stability:           {metrics['stability']:.2f}")
    
    # Print unknown detection metrics if available
    if args.use_unknown_data and unknown_metrics:
        print(f"\n{'='*60}")
        print("UNKNOWN DETECTION METRICS")
        print(f"{'='*60}")
        
        avg_f1 = np.mean([m.get('unknown_f1', 0) for m in unknown_metrics.values()])
        avg_precision = np.mean([m.get('unknown_precision', 0) for m in unknown_metrics.values()])
        avg_recall = np.mean([m.get('unknown_recall', 0) for m in unknown_metrics.values()])
        
        print(f"Average F1 Score:     {avg_f1:.3f}")
        print(f"Average Precision:    {avg_precision:.3f}")
        print(f"Average Recall:       {avg_recall:.3f}")
    
    print(f"\n{'='*60}")
    print("PER-TASK FINAL ACCURACIES")
    print(f"{'='*60}")
    
    for task_id in sorted(final_results.keys(), key=lambda x: int(x.split('_')[1])):
        if 'task_accuracy' in final_results:
            acc = final_results['task_accuracy'].get(task_id, 0)
        else:
            acc = final_results.get(task_id, 0)
        print(f"{task_id}: {acc:.2f}%")
    
    print(f"\n{'='*60}")
    print("Training completed successfully!")
    print(f"Results saved to: {args.experiment_dir}")
    print(f"{'='*60}")


def save_checkpoint_state(experiment_dir, task_idx, task_accuracies, metrics, unknown_metrics=None):
    """Save training state for resuming"""
    
    state = {
        'last_completed_task': task_idx,
        'task_accuracies': task_accuracies,
        'metrics': metrics.__dict__ if hasattr(metrics, '__dict__') else {},
        'unknown_metrics': unknown_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    state_path = os.path.join(experiment_dir, 'training_state.json')
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=4, cls=NumpyJSONEncoder)


def load_checkpoint_state(experiment_dir):
    """Load training state from checkpoint"""
    
    state_path = os.path.join(experiment_dir, 'training_state.json')
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            return json.load(f)
    return None


def run_ablation_study(args):
    """Run specific ablation study"""
    
    print(f"\n{'='*60}")
    print(f"ABLATION STUDY: {args.ablation_mode}")
    print(f"{'='*60}")
    
    if args.ablation_mode == 'no_orthogonal':
        # Test without orthogonal merging
        args.use_orthogonal_merge = False
        print("Testing without orthogonal merging...")
        
    elif args.ablation_mode == 'no_unknown':
        # Test without unknown class mechanism
        args.use_unknown_data = False
        args.lambda_task_unknown = 0.0
        args.lambda_block_unknown = 0.0
        print("Testing without unknown class mechanism...")
        
    elif args.ablation_mode == 'no_hierarchy':
        # Test without hierarchical structure
        args.use_hierarchical = False
        print("Testing without hierarchical structure...")
        
    elif args.ablation_mode == 'unknown_ratios':
        # Test different unknown ratios
        ratios = [0.1, 0.2, 0.3, 0.4]
        results = {}
        
        for ratio in ratios:
            print(f"\nTesting with unknown_ratio={ratio}...")
            args.unknown_ratio = ratio
            args.use_unknown_data = True
            args.experiment_name = f"ablation_unknown_{int(ratio*100)}"
            main(args)
            
            # Load results
            results_path = os.path.join(args.experiment_dir, 'final_results.json')
            with open(results_path, 'r') as f:
                results[f"ratio_{ratio}"] = json.load(f)
        
        # Compare results
        compare_ablation_results(results, args.save_dir)
        return
        
    elif args.ablation_mode == 'different_ranks':
        # Test different LoRA ranks
        ranks = [2, 4, 8, 16]
        results = {}
        
        for rank in ranks:
            print(f"\nTesting with rank={rank}...")
            args.lora_rank = rank
            args.experiment_name = f"ablation_rank_{rank}"
            main(args)
            
            # Load results
            results_path = os.path.join(args.experiment_dir, 'final_results.json')
            with open(results_path, 'r') as f:
                results[f"rank_{rank}"] = json.load(f)
        
        # Compare results
        compare_ablation_results(results, args.save_dir)
        return
        
    elif args.ablation_mode == 'different_blocks':
        # Test different tasks per block
        block_sizes = [3, 5, 7, 10]
        results = {}
        
        for block_size in block_sizes:
            print(f"\nTesting with {block_size} tasks per block...")
            args.max_tasks_per_block = block_size
            args.experiment_name = f"ablation_blocks_{block_size}"
            main(args)
            
            # Load results
            results_path = os.path.join(args.experiment_dir, 'final_results.json')
            with open(results_path, 'r') as f:
                results[f"blocks_{block_size}"] = json.load(f)
        
        # Compare results
        compare_ablation_results(results, args.save_dir)
        return
    
    # Run single ablation
    main(args)


def compare_ablation_results(results, save_dir):
    """Compare and visualize ablation study results"""
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract metrics
    configs = list(results.keys())
    accuracies = [r['metrics']['average_accuracy'] for r in results.values()]
    forgetting = [r['metrics']['average_forgetting'] for r in results.values()]
    
    # Plot accuracy
    ax = axes[0]
    ax.bar(configs, accuracies, color='steelblue')
    ax.set_ylabel('Average Accuracy (%)')
    ax.set_title('Ablation Study: Accuracy Comparison')
    ax.set_xticklabels(configs, rotation=45)
    
    # Plot forgetting
    ax = axes[1]
    ax.bar(configs, forgetting, color='coral')
    ax.set_ylabel('Average Forgetting (%)')
    ax.set_title('Ablation Study: Forgetting Comparison')
    ax.set_xticklabels(configs, rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_comparison.png'))
    plt.close()
    
    print(f"\nAblation results saved to: {save_dir}")


def main(args=None):
    """Main entry point"""
    
    if args is None:
        args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup experiment directory
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if args.use_hierarchical:
            unknown_suffix = "_unknown" if args.use_unknown_data else ""
            args.experiment_name = f"hierarchical_{args.lora_config}{unknown_suffix}_{timestamp}"
        else:
            args.experiment_name = f"standard_{args.lora_config}_{timestamp}"
    
    args.experiment_dir = os.path.join(args.save_dir, args.experiment_name)
    os.makedirs(args.experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(args.experiment_dir, 'checkpoints'), exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.experiment_dir)
    print(f"Experiment: {args.experiment_name}")
    
    # Save configuration
    config_path = os.path.join(args.experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4, cls=NumpyJSONEncoder)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run ablation study if specified
    if args.ablation_mode:
        run_ablation_study(args)
        return
    
    # Create model
    model = create_model(args)
    model.to(device)
    
    # Create dataset
    dataset = create_dataset(args)
    
    # Create memory buffer
    memory_buffer = MemoryBuffer(
        buffer_size=args.buffer_size,
        selection_strategy=args.selection_strategy,
        samples_per_class=args.samples_per_class
    )
    
    # Create trainer
    trainer = create_trainer(model, memory_buffer, args)
    
    # Evaluation only mode
    if args.eval_only:
        if args.eval_checkpoint:
            print(f"Loading checkpoint: {args.eval_checkpoint}")
            # Load checkpoint and evaluate
            # ... implementation ...
        else:
            logger.error("No checkpoint specified for evaluation")
        return
    
    # Run training
    results = run_training(args, model, dataset, trainer, memory_buffer, logger)
    
    print("Experiment completed successfully!")
    
    return results

def update_memory_buffer_efficient(model, memory_buffer, train_loader, task_id, args, num_batches=3):
    """
    Memory-efficient buffer update - processes data on CPU
    """
    if memory_buffer is None:
        return
    
    print(f"\n📦 Updating memory buffer with {task_id} samples...")
    
    # Collect samples batch by batch without accumulating on GPU
    all_images = []
    all_labels = []
    all_features = [] if args.selection_strategy == 'herding' else None
    
    model.eval()
    
    # Process on CPU for memory efficiency
    original_device = next(model.parameters()).device
    if args.selection_strategy == 'herding':
        model.cpu()  # Move model to CPU for feature extraction
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(train_loader):
            if batch_idx >= num_batches:
                break
            
            # Extract data
            if len(batch_data) == 3:
                images, labels, unknown_flags = batch_data
                known_mask = (unknown_flags == 0)
                if known_mask.sum() == 0:
                    continue
                images = images[known_mask]
                labels = labels[known_mask]
            else:
                images, labels = batch_data
            
            # Keep on CPU
            all_images.append(images)
            all_labels.append(labels)
            
            # Extract features on CPU if needed
            if args.selection_strategy == 'herding':
                # Process in small chunks on CPU
                chunk_size = 16
                batch_features = []
                
                for i in range(0, len(images), chunk_size):
                    chunk = images[i:i+chunk_size]  # Keep on CPU
                    feat = model.forward(chunk, task_id=task_id, return_features=True)
                    if isinstance(feat, dict):
                        feat = feat.get('features', feat)
                    batch_features.append(feat)  # Already on CPU
                    
                all_features.append(torch.cat(batch_features, dim=0))
    
    # Move model back to original device
    if args.selection_strategy == 'herding' and original_device.type == 'cuda':
        model.to(original_device)
    
    # Concatenate all CPU tensors
    if all_images:
        final_images = torch.cat(all_images, dim=0)
        final_labels = torch.cat(all_labels, dim=0)
        final_features = torch.cat(all_features, dim=0) if all_features else None
        
        # Update buffer (all on CPU)
        memory_buffer.update(
            images=final_images,
            labels=final_labels,
            task_id=task_id,
            features=final_features
        )
        
        # Print statistics
        stats = memory_buffer.get_statistics()
        print(f"  ✓ Buffer updated:")
        print(f"    - Added {len(final_images)} samples")
        print(f"    - Total: {stats['total_samples']}/{memory_buffer.buffer_size}")
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
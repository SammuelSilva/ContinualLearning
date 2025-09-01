"""
Metrics for evaluating continual learning performance.
Includes accuracy, forgetting, forward/backward transfer metrics.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from src.utils.helpers import NumpyJSONEncoder


class ContinualMetrics:
    """
    Tracks and computes metrics for continual learning experiments.
    """
    
    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks
        
        # Storage for accuracy matrices
        # accuracy_matrix[i][j] = accuracy on task j after training on task i
        self.accuracy_matrix = np.zeros((num_tasks, num_tasks))
        self.accuracy_matrix[:] = np.nan  # Initialize with NaN
        
        # Storage for per-task best accuracies
        self.best_accuracies = {}
        
        # Storage for training progress
        self.training_history = defaultdict(list)
        
        # Task order (for permuted scenarios)
        self.task_order = list(range(num_tasks))
    
    def update(self, task_trained: int, task_evaluated: int, accuracy: float):
        """
        Update accuracy matrix with new result.
        
        Args:
            task_trained: Task ID that was just trained
            task_evaluated: Task ID being evaluated
            accuracy: Accuracy achieved
        """
        self.accuracy_matrix[task_trained, task_evaluated] = accuracy
        
        # Update best accuracy for this task
        if task_evaluated not in self.best_accuracies:
            self.best_accuracies[task_evaluated] = accuracy
        else:
            self.best_accuracies[task_evaluated] = max(
                self.best_accuracies[task_evaluated], 
                accuracy
            )
    
    def compute_average_accuracy(self, after_task: Optional[int] = None) -> float:
        """
        Compute average accuracy across all tasks.
        
        Args:
            after_task: If specified, compute accuracy after training on this task
                       If None, use the last trained task
        """
        if after_task is None:
            # Find the last trained task
            after_task = self.num_tasks - 1
            while after_task >= 0 and np.all(np.isnan(self.accuracy_matrix[after_task])):
                after_task -= 1
        
        if after_task < 0:
            return 0.0
        
        # Get accuracies for all tasks up to and including after_task
        accuracies = []
        for task_id in range(after_task + 1):
            if not np.isnan(self.accuracy_matrix[after_task, task_id]):
                accuracies.append(self.accuracy_matrix[after_task, task_id])
        
        return np.mean(accuracies) if accuracies else 0.0
    
    def compute_forgetting(self, after_task: Optional[int] = None) -> Dict[str, float]:
        """
        Compute forgetting metrics.
        
        Returns:
            Dictionary with:
            - average_forgetting: Average forgetting across all tasks
            - per_task_forgetting: Forgetting for each task
            - max_forgetting: Maximum forgetting observed
        """
        if after_task is None:
            after_task = self.num_tasks - 1
            while after_task >= 0 and np.all(np.isnan(self.accuracy_matrix[after_task])):
                after_task -= 1
        
        if after_task <= 0:
            return {
                'average_forgetting': 0.0,
                'per_task_forgetting': {},
                'max_forgetting': 0.0
            }
        
        per_task_forgetting = {}
        
        for task_id in range(after_task):  # Don't compute forgetting for current task
            # Best accuracy when task was learned
            best_acc = self.accuracy_matrix[task_id, task_id]
            # Current accuracy
            current_acc = self.accuracy_matrix[after_task, task_id]
            
            if not np.isnan(best_acc) and not np.isnan(current_acc):
                forgetting = max(0, best_acc - current_acc)
                per_task_forgetting[task_id] = forgetting
        
        avg_forgetting = np.mean(list(per_task_forgetting.values())) if per_task_forgetting else 0.0
        max_forgetting = max(per_task_forgetting.values()) if per_task_forgetting else 0.0
        
        return {
            'average_forgetting': avg_forgetting,
            'per_task_forgetting': per_task_forgetting,
            'max_forgetting': max_forgetting
        }
    
    def compute_backward_transfer(self) -> float:
        """
        Compute backward transfer (BWT).
        BWT measures the influence that learning a task has on the performance 
        of previous tasks.
        
        BWT = (1/(T-1)) * Σ(R_T,i - R_i,i) for i=1 to T-1
        where R_i,j is the accuracy on task j after training on task i
        """
        T = self.num_tasks
        if T <= 1:
            return 0.0
        
        bwt_sum = 0
        count = 0
        
        for i in range(T - 1):
            # Accuracy on task i after training all tasks
            final_acc = self.accuracy_matrix[T - 1, i]
            # Accuracy on task i right after training it
            initial_acc = self.accuracy_matrix[i, i]
            
            if not np.isnan(final_acc) and not np.isnan(initial_acc):
                bwt_sum += (final_acc - initial_acc)
                count += 1
        
        return bwt_sum / count if count > 0 else 0.0
    
    def compute_forward_transfer(self) -> float:
        """
        Compute forward transfer (FWT).
        FWT measures the influence that learning a task has on the performance 
        of future tasks (zero-shot performance).
        
        FWT = (1/(T-1)) * Σ(R_i-1,i - b_i) for i=2 to T
        where b_i is the random baseline accuracy for task i
        """
        T = self.num_tasks
        if T <= 1:
            return 0.0
        
        # Assuming random baseline is 1/num_classes per task
        # For CIFAR-100 with 10 classes per task, this would be 10%
        random_baseline = 10.0
        
        fwt_sum = 0
        count = 0
        
        for i in range(1, T):
            # Zero-shot accuracy on task i (before training on it)
            if i > 0:
                zero_shot_acc = self.accuracy_matrix[i - 1, i]
                if not np.isnan(zero_shot_acc):
                    fwt_sum += (zero_shot_acc - random_baseline)
                    count += 1
        
        return fwt_sum / count if count > 0 else 0.0
    
    def compute_plasticity(self, after_task: int) -> float:
        """
        Compute plasticity - ability to learn new tasks.
        Measured as the average accuracy on new tasks when they are learned.
        """
        if after_task < 0:
            return 0.0
        
        accuracies = []
        for task_id in range(after_task + 1):
            # Accuracy on task right after learning it
            acc = self.accuracy_matrix[task_id, task_id]
            if not np.isnan(acc):
                accuracies.append(acc)
        
        return np.mean(accuracies) if accuracies else 0.0
    
    def compute_stability(self, after_task: int) -> float:
        """
        Compute stability - ability to retain old knowledge.
        Measured as 1 - normalized_forgetting.
        """
        forgetting_metrics = self.compute_forgetting(after_task)
        avg_forgetting = forgetting_metrics['average_forgetting']
        
        # Normalize forgetting (assume max possible forgetting is 100%)
        normalized_forgetting = avg_forgetting / 100.0
        
        return 1.0 - normalized_forgetting
    
    def compute_all_metrics(self, after_task: Optional[int] = None) -> Dict[str, float]:
        """
        Compute all continual learning metrics.
        """
        if after_task is None:
            after_task = self.num_tasks - 1
            while after_task >= 0 and np.all(np.isnan(self.accuracy_matrix[after_task])):
                after_task -= 1
        
        forgetting_metrics = self.compute_forgetting(after_task)
        
        return {
            'average_accuracy': self.compute_average_accuracy(after_task),
            'average_forgetting': forgetting_metrics['average_forgetting'],
            'max_forgetting': forgetting_metrics['max_forgetting'],
            'backward_transfer': self.compute_backward_transfer(),
            'forward_transfer': self.compute_forward_transfer(),
            'plasticity': self.compute_plasticity(after_task),
            'stability': self.compute_stability(after_task),
        }
    
    def plot_accuracy_matrix(self, save_path: Optional[str] = None):
        """
        Plot the accuracy matrix as a heatmap.
        """
        plt.figure(figsize=(10, 8))
        
        # Mask NaN values
        mask = np.isnan(self.accuracy_matrix)
        
        sns.heatmap(
            self.accuracy_matrix,
            mask=mask,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            vmin=0,
            vmax=100,
            cbar_kws={'label': 'Accuracy (%)'},
            xticklabels=[f'Task {i}' for i in range(self.num_tasks)],
            yticklabels=[f'After Task {i}' for i in range(self.num_tasks)]
        )
        
        plt.title('Continual Learning Accuracy Matrix')
        plt.xlabel('Task Evaluated')
        plt.ylabel('Training Progress')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_metrics_evolution(self, save_path: Optional[str] = None):
        """
        Plot the evolution of metrics across tasks.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        tasks = list(range(self.num_tasks))
        
        # 1. Average Accuracy Evolution
        ax = axes[0, 0]
        avg_accs = [self.compute_average_accuracy(t) for t in tasks]
        ax.plot(tasks, avg_accs, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Task')
        ax.set_ylabel('Average Accuracy (%)')
        ax.set_title('Average Accuracy Evolution')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        # 2. Forgetting Evolution
        ax = axes[0, 1]
        forgettings = [self.compute_forgetting(t)['average_forgetting'] for t in tasks]
        ax.plot(tasks, forgettings, marker='s', linewidth=2, markersize=8, color='coral')
        ax.set_xlabel('Task')
        ax.set_ylabel('Average Forgetting (%)')
        ax.set_title('Forgetting Evolution')
        ax.grid(True, alpha=0.3)
        
        # 3. Plasticity vs Stability
        ax = axes[1, 0]
        plasticities = [self.compute_plasticity(t) for t in tasks]
        stabilities = [self.compute_stability(t) * 100 for t in tasks]  # Convert to percentage
        ax.plot(tasks, plasticities, marker='o', label='Plasticity', linewidth=2, markersize=8)
        ax.plot(tasks, stabilities, marker='s', label='Stability', linewidth=2, markersize=8)
        ax.set_xlabel('Task')
        ax.set_ylabel('Score (%)')
        ax.set_title('Plasticity vs Stability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        # 4. Per-Task Final Accuracy
        ax = axes[1, 1]
        final_task = self.num_tasks - 1
        final_accs = []
        for task_id in range(self.num_tasks):
            acc = self.accuracy_matrix[final_task, task_id]
            final_accs.append(acc if not np.isnan(acc) else 0)
        
        bars = ax.bar(tasks, final_accs, color='steelblue', alpha=0.8)
        ax.set_xlabel('Task')
        ax.set_ylabel('Final Accuracy (%)')
        ax.set_title('Per-Task Final Accuracy')
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, acc in zip(bars, final_accs):
            if acc > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{acc:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Continual Learning Metrics', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def get_summary(self) -> str:
        """
        Get a formatted string summary of all metrics.
        """
        metrics = self.compute_all_metrics()
        
        summary = "=" * 50 + "\n"
        summary += "CONTINUAL LEARNING METRICS SUMMARY\n"
        summary += "=" * 50 + "\n"
        summary += f"Average Accuracy:     {metrics['average_accuracy']}%\n"
        summary += f"Average Forgetting:   {metrics['average_forgetting']}%\n"
        summary += f"Max Forgetting:       {metrics['max_forgetting']}%\n"
        summary += f"Backward Transfer:    {metrics['backward_transfer']}%\n"
        summary += f"Forward Transfer:     {metrics['forward_transfer']}%\n"
        summary += f"Plasticity:          {metrics['plasticity']}%\n"
        summary += f"Stability:           {metrics['stability']}\n"
        summary += "=" * 50
        
        return summary
    
    def save_metrics(self, save_path: str):
        """
        Save all metrics to a JSON file.
        """
        import json
        
        metrics = self.compute_all_metrics()
        
        # Convert numpy arrays to lists for JSON serialization
        data = {
            'summary': metrics,
            'accuracy_matrix': self.accuracy_matrix.tolist(),
            'per_task_forgetting': self.compute_forgetting()['per_task_forgetting'],
            'best_accuracies': self.best_accuracies,
            'task_order': self.task_order
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=4, cls=NumpyJSONEncoder)
        
        print(f"Metrics saved to: {save_path}")
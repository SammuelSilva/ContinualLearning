"""
Training logic for hierarchical LoRA with dual unknown class mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from src.trainers.continual_trainer import ContinualTrainer
from src.models.hierarchical_lora import HierarchicalLoRAViT
from src.utils.visualization import HierarchicalVisualizer


class HierarchicalTrainer(ContinualTrainer):
    """
    Extended trainer for hierarchical LoRA with dual unknown mechanism.
    """
    
    def __init__(
        self,
        model: HierarchicalLoRAViT,
        memory_buffer,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        lambda_task_unknown: float = 0.5,
        lambda_block_unknown: float = 0.3,
        save_dir: str = "./checkpoints",
        num_tasks: int = 10
    ):
        super().__init__(
            model=model,
            memory_buffer=memory_buffer,
            device=device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lambda_unknown=lambda_task_unknown,
            save_dir=save_dir,
            num_tasks=num_tasks
        )
        
        self.lambda_block_unknown = lambda_block_unknown
        self.visualizer = HierarchicalVisualizer(save_dir)
        
        # Track block-level metrics
        self.block_metrics = {
            'block_accuracy': [],
            'block_confusion': [],
            'orthogonality_scores': []
        }
        if torch.cuda.is_available():
            self.to(torch.device('cuda:0'))

    def compute_hierarchical_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: str,
        memory_batch: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss with dual unknown mechanism.
        
        Includes:
        1. Task-level unknown loss
        2. Block-level unknown loss
        3. Current task classification loss
        """
        losses = {}
        
        # Get current block info
        block_id = self.model.current_block_id
        is_first_task_in_block = len(self.model.active_block_tasks) == 1
        
        # Forward pass
        features = self.model.backbone(x)
        
        # 1. Current task classification loss
        task_logits = self.model.task_heads[task_id](features)
        task_logits_no_unknown = task_logits[:, :-1]
        loss_classification = F.cross_entropy(task_logits_no_unknown, y)
        losses['classification'] = loss_classification
        
        # 2. Task-level unknown loss (within block)
        if memory_batch is not None and len(memory_batch['images']) > 0:
            mem_features = self.model.backbone(memory_batch['images'])
            mem_task_logits = self.model.task_heads[task_id](mem_features)
            
            # Label memory samples as task-unknown
            task_unknown_labels = torch.full(
                (len(memory_batch['images']),),
                self.model.task_classes[task_id],  # Last index is unknown
                dtype=torch.long,
                device=self.device
            )
            
            loss_task_unknown = F.cross_entropy(mem_task_logits, task_unknown_labels)
            losses['task_unknown'] = loss_task_unknown
        
        # 3. Block-level unknown loss (between blocks)
        if len(self.model.merged_blocks) > 0 and memory_batch is not None:
            # Sample from previous blocks
            prev_block_samples = self._sample_from_previous_blocks(batch_size=16)
            
            if prev_block_samples is not None:
                prev_features = self.model.backbone(prev_block_samples)
                
                # Current block should recognize these as "not in block"
                block_head = self.model.block_unknown_heads[block_id]
                block_logits = block_head(prev_features)
                
                # Label as "unknown" to current block (index 1)
                block_unknown_labels = torch.ones(
                    len(prev_block_samples),
                    dtype=torch.long,
                    device=self.device
                )
                
                loss_block_unknown = F.cross_entropy(block_logits, block_unknown_labels)
                losses['block_unknown'] = loss_block_unknown
        
        # 4. Intra-block regularization (keep tasks in same block similar)
        if len(self.model.active_block_tasks) > 1:
            loss_intra_block = self._compute_intra_block_regularization(features, task_id)
            losses['intra_block_reg'] = loss_intra_block
        
        # Combine losses
        total_loss = losses['classification']
        if 'task_unknown' in losses:
            total_loss += self.lambda_unknown * losses['task_unknown']
        if 'block_unknown' in losses:
            total_loss += self.lambda_block_unknown * losses['block_unknown']
        if 'intra_block_reg' in losses:
            total_loss += 0.1 * losses['intra_block_reg']
        
        losses['total'] = total_loss
        return losses
    
    def _sample_from_previous_blocks(self, batch_size: int) -> Optional[torch.Tensor]:
        """
        Sample data from previous merged blocks.
        """
        if len(self.model.merged_blocks) == 0:
            return None
        
        # Get samples from memory buffer that belong to previous blocks
        all_samples = []
        
        for block in self.model.merged_blocks:
            for task_id in block.task_ids:
                # Get samples from this task
                task_samples = self.memory_buffer.sample(
                    batch_size=batch_size // len(self.model.merged_blocks),
                    exclude_task=None  # Include all tasks
                )
                
                if len(task_samples['images']) > 0:
                    all_samples.append(task_samples['images'])
        
        if all_samples:
            return torch.cat(all_samples, dim=0)[:batch_size]
        return None
    
    def _compute_intra_block_regularization(
        self,
        features: torch.Tensor,
        current_task: str
    ) -> torch.Tensor:
        """
        Regularization to keep tasks within the same block similar.
        """
        if len(self.model.active_block_tasks) <= 1:
            return torch.tensor(0.0, device=self.device)
        
        # Get other tasks in the same block
        other_tasks = [
            task_id for task_id in self.model.active_block_tasks.keys()
            if task_id != current_task
        ]
        
        if not other_tasks:
            return torch.tensor(0.0, device=self.device)
        
        # Compute feature similarity between tasks
        current_features = features.mean(dim=0)  # Average pooling
        
        similarity_loss = 0.0
        for other_task in other_tasks:
            # Get some samples from the other task
            other_samples = self.memory_buffer.sample(batch_size=8)
            
            if len(other_samples['images']) > 0:
                other_features = self.model.backbone(
                    other_samples['images'].to(self.device)
                ).mean(dim=0)
                
                # Encourage similarity (minimize negative cosine similarity)
                similarity = F.cosine_similarity(
                    current_features.unsqueeze(0),
                    other_features.unsqueeze(0)
                )
                similarity_loss += (1 - similarity).mean()
        
        return similarity_loss / len(other_tasks)
    
    def evaluate_hierarchical(
        self,
        test_loaders: Dict[str, DataLoader]
    ) -> Dict:
        """
        Evaluate with hierarchical task prediction.
        """
        self.model.eval()
        results = {
            'task_accuracy': {},
            'block_accuracy': {},
            'confusion_matrix': np.zeros((len(test_loaders), len(test_loaders)))
        }
        
        for true_task_id, test_loader in test_loaders.items():
            correct = 0
            total = 0
            block_correct = 0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Hierarchical prediction
                    predicted_tasks, confidences = self.model.predict_task_id_hierarchical(images)
                    
                    # Task-level accuracy
                    for i, (pred_task, label) in enumerate(zip(predicted_tasks, labels)):
                        if pred_task == true_task_id:
                            # Check if class prediction is correct
                            task_logits = self.model.forward(
                                images[i:i+1],
                                task_id=pred_task
                            )
                            pred_class = torch.argmax(task_logits[:, :-1], dim=1)
                            if pred_class == label:
                                correct += 1
                        
                        total += 1
                        
                        # Update confusion matrix
                        true_idx = int(true_task_id.split('_')[1])
                        pred_idx = int(pred_task.split('_')[1])
                        results['confusion_matrix'][true_idx, pred_idx] += 1
            
            accuracy = 100 * correct / total
            results['task_accuracy'][true_task_id] = accuracy
            
            print(f"Task {true_task_id}: {accuracy:.2f}% (Hierarchical)")
        
        # Compute block-level metrics
        results['avg_task_accuracy'] = np.mean(list(results['task_accuracy'].values()))
        
        # Visualize hierarchy
        self.model.visualize_hierarchy()
        
        return results
    
    def train_task(
        self,
        task_id: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 20,
        patience: int = 5,
        task_idx: Optional[int] = None
    ):
        """
        Extended training with hierarchical features.
        """
        print(f"\n{'='*60}")
        print(f"Training Task {task_id} (Hierarchical Mode)")
        print(f"Current Block: {self.model.current_block_id}")
        print(f"Tasks in Block: {len(self.model.active_block_tasks)}")
        print(f"{'='*60}\n")
        
        # Set active task
        self.model.set_active_task(task_id)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.model.get_trainable_parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler with warmup for new blocks
        if len(self.model.active_block_tasks) == 1:
            # First task in new block - use warmup
            warmup_steps = num_epochs // 5
            scheduler = self._create_warmup_scheduler(optimizer, warmup_steps, num_epochs)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs
            )
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self._train_epoch_hierarchical(
                train_loader, optimizer, task_id
            )
            
            # Validation
            val_loss, val_acc = self._validate(val_loader, task_id)
            
            # Update scheduler
            scheduler.step()
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self._save_checkpoint(task_id, epoch, val_acc)
            else:
                patience_counter += 1
            
            # Log metrics
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Update memory buffer
        self._update_memory_buffer(train_loader, task_id)
        
        # Compute and store orthogonality score
        if len(self.model.active_block_tasks) > 1:
            from src.models.orthogonal_utils import compute_orthogonality_score
            
            adapters = [
                self.model.task_adapters[tid]
                for tid in self.model.active_block_tasks.keys()
            ]
            ortho_score = compute_orthogonality_score(adapters)
            self.block_metrics['orthogonality_scores'].append(ortho_score)
            print(f"\nBlock Orthogonality Score: {ortho_score:.3f}")
        
        # Visualize if end of block
        if len(self.model.active_block_tasks) == self.model.tasks_per_block:
            self._visualize_block_completion()
        
            from src.utils.visualization import HierarchicalVisualizer, MetricsAnimator
    
        visualizer = HierarchicalVisualizer(save_dir=self.save_dir)
        
        # Plot hierarchy tree
        hierarchy_stats = self.model.get_statistics()
        visualizer.plot_hierarchy_tree(
            hierarchy_stats,
            save_path=f"{self.save_dir}/hierarchy_task_{task_idx}.html"
        )
    
        # Plot memory efficiency
        visualizer.plot_memory_efficiency(
            num_tasks=task_idx + 1,
            tasks_per_block=self.model.tasks_per_block,
            lora_rank=self.model.lora_rank,
            save_path=f"{self.save_dir}/memory_efficiency_task_{task_idx}.png"
        )
        
        # Plot orthogonality matrix if we have multiple tasks
        if len(self.model.task_adapters) > 1:
            visualizer.plot_orthogonality_matrix(
                self.model.task_adapters,
                save_path=f"{self.save_dir}/orthogonality_task_{task_idx}.png"
            )
        
        # Create training dashboard at the end
        if task_idx == self.num_tasks - 1:
            visualizer.create_training_dashboard(
                metrics_history=self.metrics.training_history,
                hierarchy_stats=hierarchy_stats,
                save_path=f"{self.save_dir}/final_dashboard.html"
            )
            
            # Create accuracy animation
            animator = MetricsAnimator()
            animator.create_accuracy_animation(
                self.metrics.accuracy_matrix,
                save_path=f"{self.save_dir}/accuracy_evolution.gif"
            )
        
        return best_val_acc

    
    def _train_epoch_hierarchical(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        task_id: str
    ) -> Tuple[float, float]:
        """
        Training epoch with hierarchical loss.
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training (Hierarchical)")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Sample from memory buffer
            memory_batch = None
            if len(self.memory_buffer) > 0:
                memory_batch = self.memory_buffer.sample(
                    batch_size=min(32, len(images)),
                    exclude_task=task_id
                )
                if len(memory_batch['images']) > 0:
                    memory_batch['images'] = memory_batch['images'].to(self.device)
            
            # Compute hierarchical loss
            losses = self.compute_hierarchical_loss(
                images, labels, task_id, memory_batch
            )
            
            # Backward pass
            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.get_trainable_parameters(), 1.0
            )
            optimizer.step()
            
            # Update metrics
            with torch.no_grad():
                logits = self.model(images, task_id=task_id)
                predictions = torch.argmax(logits[:, :-1], dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                total_loss += losses['total'].item()
            
            # Update progress bar
            loss_str = f"L: {losses['total'].item():.3f}"
            if 'block_unknown' in losses:
                loss_str += f" | BU: {losses['block_unknown'].item():.3f}"
            
            pbar.set_postfix_str(f"{loss_str} | Acc: {100*correct/total:.1f}%")
        
        return total_loss / len(train_loader), 100 * correct / total
    
    def _create_warmup_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int
    ):
        """
        Create a scheduler with linear warmup.
        """
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_steps - step) / float(max(1, total_steps - warmup_steps))
            )
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def _visualize_block_completion(self):
        """
        Visualize metrics when a block is completed.
        """
        print("\n" + "="*60)
        print(f"Block {self.model.current_block_id} Completed!")
        print("="*60)
        
        # Show orthogonality scores
        if self.block_metrics['orthogonality_scores']:
            scores = self.block_metrics['orthogonality_scores'][-5:]  # Last 5
            print(f"Orthogonality Scores: {scores}")
            print(f"Average: {np.mean(scores):.3f}")
        
        # Visualize structure
        stats = self.model.get_statistics()
        save_path = f"{self.save_dir}/hierarchy_block_{self.model.current_block_id}.html"
        self.visualizer.plot_hierarchy_tree(stats, save_path)
        print(f"Hierarchy visualization saved to: {save_path}")
    
    def _validate(self, val_loader: DataLoader, task_id: str) -> Tuple[float, float]:
        """
        Validate on validation set (inherited from parent but needed for reference).
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(images, task_id=task_id)
                
                # Compute loss (only current task)
                loss = F.cross_entropy(logits[:, :-1], labels)
                total_loss += loss.item()
                
                # Compute accuracy
                predictions = torch.argmax(logits[:, :-1], dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy

    def _save_checkpoint(self, task_id: str, epoch: int, val_acc: float):
        """
        Save model checkpoint (inherited from parent).
        """
        checkpoint_path = os.path.join(
            self.save_dir,
            f"task_{task_id.split('_')[1]}_epoch{epoch}_acc{val_acc:.2f}.pt"
        )
        
        self.model.save_task_checkpoint(task_id, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def _update_memory_buffer(self, train_loader: DataLoader, task_id: str):
        """
        Update memory buffer with task data (inherited from parent).
        """
        print("Updating memory buffer...")
        
        all_images = []
        all_labels = []
        all_features = []
        
        self.model.eval()
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(self.device)
                
                # Get features
                features = self.model.forward(images, task_id=task_id, return_features=True)
                
                all_images.append(images.cpu())
                all_labels.append(labels)
                all_features.append(features.cpu())
        
        # Concatenate all data
        all_images = torch.cat(all_images, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_features = torch.cat(all_features, dim=0)
        
        # Update buffer
        self.memory_buffer.update(
            all_images, all_labels, task_id, all_features
        )
        
        print(f"Buffer updated. Current size: {len(self.memory_buffer)}")

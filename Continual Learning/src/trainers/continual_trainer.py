"""
Main training logic for continual learning with LoRA-ViT
"""

import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
from src.models.lora_vit import ContinualLoRAViT
from src.data.memory_buffer import MemoryBuffer
from src.utils.metrics import ContinualMetrics


from typing import List, Tuple, Dict, Optional


class ContinualTrainer:
    """
    Trainer for continual learning with LoRA-ViT model.
    Implements the training loop with unknown class mechanism.
    """
    
    def __init__(
        self,
        model: ContinualLoRAViT,
        memory_buffer: MemoryBuffer,
        device: torch.device = torch.device("cuda"),
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        lambda_unknown: float = 0.5,
        num_tasks: int = 10,
        save_dir: str = "./checkpoints"
    ):
        self.model = model.to(device)
        self.memory_buffer = memory_buffer
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lambda_unknown = lambda_unknown
        self.save_dir = save_dir
        
        # Metrics tracker
        self.metrics = ContinualMetrics(num_tasks)
        self.current_task_idx = 0

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
    
    def train_task(
        self,
        task_id: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 20,
        patience: int = 5,
        task_idx: int = None
    ):
        """Train model on a single task"""
        
        print(f"\n=== Training Task {task_id} ===")
        
        if task_idx is not None:
            self.current_task_idx = task_idx

        # Set active task
        self.model.set_active_task(task_id)
        
        # Create optimizer for current task only
        optimizer = torch.optim.AdamW(
            self.model.get_trainable_parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
        
        # Early stopping
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(
                train_loader, optimizer, task_id
            )
            
            # Validation phase
            val_loss, val_acc = self._validate(val_loader, task_id)
            
            # Update scheduler
            scheduler.step()
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best checkpoint
                self._save_checkpoint(task_id, epoch, val_acc)
            else:
                patience_counter += 1
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Update memory buffer with task data
        self._update_memory_buffer(train_loader, task_id)
        
        # Freeze task parameters
        self.model.freeze_previous_tasks()
        
        return best_val_acc
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        task_id: str
    ) -> Tuple[float, float]:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Sample from memory buffer
            memory_batch = None
            if len(self.memory_buffer) > 0:
                memory_batch = self.memory_buffer.sample(
                    batch_size=32,
                    exclude_task=task_id
                )
                if len(memory_batch['images']) > 0:
                    memory_batch['images'] = memory_batch['images'].to(self.device)
            
            # Compute loss
            losses = self.model.compute_loss(
                images, labels, task_id,
                memory_buffer=memory_batch,
                lambda_unknown=self.lambda_unknown
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
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'acc': 100 * correct / total
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def _validate(
        self,
        val_loader: DataLoader,
        task_id: str
    ) -> Tuple[float, float]:
        """Validate on validation set"""
        
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
                loss = torch.nn.functional.cross_entropy(logits[:, :-1], labels)
                total_loss += loss.item()
                
                # Compute accuracy
                predictions = torch.argmax(logits[:, :-1], dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def _update_memory_buffer(
        self,
        train_loader: DataLoader,
        task_id: str
    ):
        """Update memory buffer with task data"""
        
        print("Updating memory buffer...")
        
        # Collect all task data
        all_images = []
        all_labels = []
        all_features = []
        
        self.model.eval()
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(self.device)
                
                # Get features for herding selection
                features = self.model.forward_features_with_lora(
                    images, task_id=task_id
                )
                
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
        print(f"Buffer statistics: {self.memory_buffer.get_statistics()}")
    
    def _save_checkpoint(
        self,
        task_id: str,
        epoch: int,
        val_acc: float
    ):
        """Save model checkpoint"""
        
        checkpoint_path = os.path.join(
            self.save_dir,
            f"{task_id}_epoch{epoch}_acc{val_acc:.2f}.pt"
        )
        
        self.model.save_task_checkpoint(task_id, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def evaluate_all_tasks(
        self,
        test_loaders: Dict[str, DataLoader]
    ) -> Dict[str, float]:
        """Evaluate model on all tasks"""
        
        self.model.eval()
        results = {}
        
        for task_id, test_loader in test_loaders.items():
            correct = 0
            total = 0
            print(f"DEBUG: Continual Trainer Evaluating task_{task_id}")
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Predict with task ID inference
                    predictions, predicted_tasks = self.model.predict(images)
                    
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
            
            accuracy = 100 * correct / total
            results[task_id] = accuracy
            print(f"Task {task_id}: {accuracy:.2f}%")
        
        # Compute average accuracy
        avg_accuracy = np.mean(list(results.values()))
        all_metrics = self.metrics.compute_all_metrics(after_task=self.current_task_idx)
        print(f"Average Accuracy: {avg_accuracy:.2f}%")
        print(f"\n--- Continual Learning Metrics ---")
        print(f"Average Accuracy: {all_metrics['average_accuracy']:.2f}%")
        print(f"Average Forgetting: {all_metrics['average_forgetting']:.2f}%")
        print(f"Backward Transfer: {all_metrics['backward_transfer']:.2f}%")
        print(f"Forward Transfer: {all_metrics['forward_transfer']:.2f}%")
        print(f"Plasticity: {all_metrics['plasticity']:.2f}%")
        print(f"Stability: {all_metrics['stability']:.2f}")
        
        return results
    
    def get_metrics_summary(self) -> Dict:
        """Get comprehensive metrics summary"""
        return self.metrics.compute_all_metrics()
    
    def save_metrics(self, save_path: str):
        """Save metrics to file"""
        self.metrics.save_metrics(save_path)
    
    def plot_metrics(self, save_dir: str):
        """Generate and save metric plots"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot accuracy matrix
        matrix_path = os.path.join(save_dir, 'accuracy_matrix.png')
        self.metrics.plot_accuracy_matrix(matrix_path)
        
        # Plot metrics evolution
        evolution_path = os.path.join(save_dir, 'metrics_evolution.png')
        self.metrics.plot_metrics_evolution(evolution_path)
        
        print(f"Metric plots saved to {save_dir}")

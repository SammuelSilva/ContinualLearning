"""
Enhanced Hierarchical Trainer with Unknown Sample Handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import logging


class HierarchicalTrainer:
    """
    Trainer for Hierarchical LoRA-ViT with enhanced unknown sample handling.
    """
    
    def __init__(
        self,
        model: nn.Module,
        memory_buffer: Optional[object] = None,
        device: torch.device = torch.device('cuda'),
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        lambda_task_unknown: float = 0.5,
        lambda_block_unknown: float = 0.3,
        lambda_classification: float = 1.0,
        unknown_temperature: float = 2.0,
        save_dir: str = './checkpoints',
        num_tasks: int = 10
    ):
        self.model = model
        self.memory_buffer = memory_buffer
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lambda_task_unknown = lambda_task_unknown
        self.lambda_block_unknown = lambda_block_unknown
        self.lambda_classification = lambda_classification
        self.unknown_temperature = unknown_temperature
        self.save_dir = save_dir
        self.num_tasks = num_tasks
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize metrics tracking
        self.metrics = self._initialize_metrics()
        
    def _initialize_metrics(self):
        """Initialize metrics tracking"""
        try:
            from src.utils.metrics import ContinualMetrics
            return ContinualMetrics(num_tasks=self.num_tasks)
        except ImportError:
            # Fallback if metrics module is not available
            return None
    
    def train_task(
        self,
        task_id: str,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 20,
        patience: int = 5,
        task_idx: int = 0,
        warmup_epochs: int = 2
    ) -> float:
        """
        Train on a specific task with unknown sample handling.
        """
        
        self.logger.info(f"Training {task_id} (Task {task_idx})")
        
        # Setup optimizer
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Setup scheduler with warmup for early tasks
        if task_idx == 0:
            total_steps = len(train_loader) * (num_epochs + warmup_epochs)
            warmup_steps = len(train_loader) * warmup_epochs
        else:
            total_steps = len(train_loader) * num_epochs
            warmup_steps = 0
        
        scheduler = self._get_scheduler(optimizer, total_steps, warmup_steps)
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(num_epochs + (warmup_epochs if task_idx == 0 else 0)):
            # Training phase
            train_loss, train_metrics = self._train_epoch(
                task_id, train_loader, optimizer, scheduler, task_idx, epoch
            )
            
            # Validation phase
            val_loss, val_metrics = self._validate(task_id, val_loader, task_idx)
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs + (warmup_epochs if task_idx == 0 else 0)}: "
                f"Train Loss: {train_loss:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                f"Train Unknown F1: {train_metrics.get('unknown_f1', -1):.3f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                f"Val Unknown F1: {val_metrics.get('unknown_f1', -1):.3f}"
            )
            
            # Early stopping
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
                # Save best model
                self._save_checkpoint(task_id, epoch, best_val_acc)
            else:
                patience_counter += 1
                if patience_counter >= patience and epoch > warmup_epochs:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        return best_val_acc
    
    def _train_epoch(
        self,
        task_id: str,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        task_idx: int,
        epoch: int
    ) -> Tuple[float, Dict]:
        """Train for one epoch with unknown sample handling"""
        
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        unknown_tp = 0
        unknown_fp = 0
        unknown_tn = 0
        unknown_fn = 0
        
        pbar = tqdm(train_loader, desc=f"Training epoch {epoch+1}")
        
        for batch_idx, batch_data in enumerate(pbar):
            # Handle different batch formats
            if len(batch_data) == 3:
                images, labels, unknown_flags = batch_data
                images = images.to(self.device)
                labels = labels.to(self.device)
                unknown_flags = unknown_flags.to(self.device).float()  # Ensure float
                has_unknown = True
            else:
                images, labels = batch_data
                images = images.to(self.device)
                labels = labels.to(self.device)
                unknown_flags = torch.zeros_like(labels, dtype=torch.float)  # Explicit float type
                has_unknown = False
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images, task_id=task_id)
            
            # Handle case where model returns tensor directly instead of dict
            if isinstance(outputs, torch.Tensor):
                # Model returned logits directly, wrap in expected format
                logits = outputs
                outputs = {'logits': logits}
            elif isinstance(outputs, dict):
                # Model returned dictionary as expected
                if 'logits' not in outputs:
                    raise ValueError("Model output dictionary must contain 'logits' key")
                logits = outputs['logits']
            else:
                raise ValueError(f"Expected model output to be torch.Tensor or dict, got {type(outputs)}")
            
            if logits.dim() != 2:
                raise ValueError(f"Expected logits to be 2D tensor [batch_size, num_classes], got shape {logits.shape}")
            
            # Compute losses
            loss, loss_components = self._compute_loss(
                outputs, labels, unknown_flags, task_id, task_idx, has_unknown
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Calculate accuracy (excluding unknown samples for classification accuracy)
            with torch.no_grad():
                if has_unknown:
                    known_mask = (unknown_flags == 0)
                    if known_mask.sum() > 0:
                        known_logits = logits[known_mask]  # Shape: [num_known, num_classes]
                        known_labels = labels[known_mask]   # Shape: [num_known]
                        _, predicted = known_logits.max(1)
                        correct += predicted.eq(known_labels).sum().item()
                        total += known_mask.sum().item()
                    
                    # Calculate unknown detection metrics
                    if 'unknown_score' in outputs:
                        unknown_scores = outputs['unknown_score'].squeeze()
                        if unknown_scores.dim() == 0:  # Handle single sample case
                            unknown_scores = unknown_scores.unsqueeze(0)
                        
                        unknown_pred = (torch.sigmoid(unknown_scores) > 0.5).float()
                        
                        # Ensure same shape for comparison
                        if unknown_pred.shape != unknown_flags.shape:
                            if unknown_flags.dim() > unknown_pred.dim():
                                unknown_pred = unknown_pred.squeeze()
                            elif unknown_pred.dim() > unknown_flags.dim():
                                unknown_flags = unknown_flags.squeeze()
                        
                        unknown_tp += ((unknown_pred == 1) & (unknown_flags == 1)).sum().item()
                        unknown_fp += ((unknown_pred == 1) & (unknown_flags == 0)).sum().item()
                        unknown_tn += ((unknown_pred == 0) & (unknown_flags == 0)).sum().item()
                        unknown_fn += ((unknown_pred == 0) & (unknown_flags == 1)).sum().item()
                else:
                    _, predicted = logits.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.1f}%' if total > 0 else '0.0%',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        # Calculate metrics
        accuracy = 100. * correct / total if total > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'loss': total_loss / len(train_loader)
        }
        
        # Add unknown detection metrics if applicable
        if has_unknown and (unknown_tp + unknown_fp + unknown_tn + unknown_fn) > 0:
            precision = unknown_tp / (unknown_tp + unknown_fp) if (unknown_tp + unknown_fp) > 0 else 0
            recall = unknown_tp / (unknown_tp + unknown_fn) if (unknown_tp + unknown_fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics.update({
                'unknown_precision': precision,
                'unknown_recall': recall,
                'unknown_f1': f1,
                'unknown_accuracy': (unknown_tp + unknown_tn) / (unknown_tp + unknown_fp + unknown_tn + unknown_fn)
            })
        
        return total_loss / len(train_loader), metrics
    
    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        unknown_flags: torch.Tensor,
        task_id: str,
        task_idx: int,
        has_unknown: bool
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss with unknown handling.
        """
        
        loss_components = {}
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Classification loss (only for known samples)
        if has_unknown:
            known_mask = (unknown_flags == 0)
            if known_mask.sum() > 0:
                # FIX: Use boolean indexing correctly for 2D tensor
                known_logits = outputs['logits'][known_mask]  # Shape: [num_known_samples, num_classes]
                known_labels = labels[known_mask]              # Shape: [num_known_samples]
                
                classification_loss = F.cross_entropy(known_logits, known_labels)
                loss_components['classification'] = classification_loss.item()
                total_loss = total_loss + self.lambda_classification * classification_loss
        else:
            classification_loss = F.cross_entropy(outputs['logits'], labels)
            loss_components['classification'] = classification_loss.item()
            total_loss = total_loss + self.lambda_classification * classification_loss
        
        # Unknown detection loss
        if has_unknown and 'unknown_score' in outputs:
            unknown_loss = F.binary_cross_entropy_with_logits(
                outputs['unknown_score'].squeeze(),
                unknown_flags.float(),  # Ensure float type
                pos_weight=torch.tensor([2.0]).to(self.device)
            )
            loss_components['unknown'] = unknown_loss.item()
            
            # Scale unknown loss based on task
            unknown_weight = self.lambda_task_unknown * (0.9 ** task_idx)
            total_loss = total_loss + unknown_weight * unknown_loss
        
        # Task-level unknown regularization
        if 'task_unknown_score' in outputs and task_idx > 0:
            task_unknown_loss = self._compute_task_unknown_loss(
                outputs['task_unknown_score'],
                task_id,
                labels,
                unknown_flags
            )
            loss_components['task_unknown'] = task_unknown_loss.item()
            total_loss = total_loss + self.lambda_task_unknown * task_unknown_loss
        
        # Block-level unknown regularization
        if 'block_unknown_score' in outputs:
            block_unknown_loss = self._compute_block_unknown_loss(
                outputs['block_unknown_score'],
                task_id,
                labels,
                unknown_flags
            )
            loss_components['block_unknown'] = block_unknown_loss.item()
            total_loss = total_loss + self.lambda_block_unknown * block_unknown_loss
        
        # Orthogonality regularization
        if 'orthogonal_loss' in outputs:
            loss_components['orthogonal'] = outputs['orthogonal_loss'].item()
            total_loss = total_loss + 0.1 * outputs['orthogonal_loss']
        
        return total_loss, loss_components

    def _compute_task_unknown_loss(
        self,
        task_unknown_scores: torch.Tensor,
        task_id: str,
        labels: torch.Tensor,
        unknown_flags: torch.Tensor
    ) -> torch.Tensor:
        """Compute task-level unknown detection loss"""
        
        # For current task samples, should predict "known" (0)
        # For unknown samples, should predict "unknown" (1)
        
        # Create target: 0 for current task samples, 1 for others
        target = unknown_flags.clone()
        
        # Use temperature scaling for smoother gradients
        scaled_scores = task_unknown_scores / self.unknown_temperature
        
        return F.binary_cross_entropy_with_logits(
            scaled_scores.squeeze(),
            target
        )
    
    def _compute_block_unknown_loss(
        self,
        block_unknown_scores: torch.Tensor,
        task_id: str,
        labels: torch.Tensor,
        unknown_flags: torch.Tensor
    ) -> torch.Tensor:
        """Compute block-level unknown detection loss"""
        
        # Similar to task unknown but at block level
        target = unknown_flags.clone()
        
        scaled_scores = block_unknown_scores / self.unknown_temperature
        
        return F.binary_cross_entropy_with_logits(
            scaled_scores.squeeze(),
            target
        )
    
    def _validate(
        self,
        task_id: str,
        val_loader: torch.utils.data.DataLoader,
        task_idx: int
    ) -> Tuple[float, Dict]:
        """Validate on a specific task"""
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        unknown_tp = 0
        unknown_fp = 0
        unknown_tn = 0
        unknown_fn = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                # Handle different batch formats
                if len(batch_data) == 3:
                    images, labels, unknown_flags = batch_data
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    unknown_flags = unknown_flags.to(self.device).float()
                    has_unknown = True
                else:
                    images, labels = batch_data
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    unknown_flags = torch.zeros_like(labels, dtype=torch.float)
                    has_unknown = False
                
                outputs = self.model(images, task_id=task_id)
                
                # Handle case where model returns tensor directly instead of dict
                if isinstance(outputs, torch.Tensor):
                    # Model returned logits directly, wrap in expected format
                    logits = outputs
                    outputs = {'logits': logits}
                elif isinstance(outputs, dict):
                    # Model returned dictionary as expected
                    if 'logits' not in outputs:
                        raise ValueError("Model output dictionary must contain 'logits' key")
                    logits = outputs['logits']
                else:
                    raise ValueError(f"Expected model output to be torch.Tensor or dict, got {type(outputs)}")
                
                if logits.dim() != 2:
                    raise ValueError(f"Expected logits to be 2D tensor, got shape {logits.shape}")
                
                # Compute loss
                loss, _ = self._compute_loss(
                    outputs, labels, unknown_flags, task_id, task_idx, has_unknown
                )
                total_loss += loss.item()
                
                # Calculate accuracy
                if has_unknown:
                    known_mask = (unknown_flags == 0)
                    if known_mask.sum() > 0:
                        known_logits = logits[known_mask]
                        known_labels = labels[known_mask]
                        _, predicted = known_logits.max(1)
                        correct += predicted.eq(known_labels).sum().item()
                        total += known_mask.sum().item()
                    
                    # Calculate unknown detection metrics
                    if 'unknown_score' in outputs:
                        unknown_scores = outputs['unknown_score'].squeeze()
                        if unknown_scores.dim() == 0:
                            unknown_scores = unknown_scores.unsqueeze(0)
                        
                        unknown_pred = (torch.sigmoid(unknown_scores) > 0.5).float()
                        
                        # Ensure compatible shapes
                        if unknown_pred.shape != unknown_flags.shape:
                            if unknown_flags.dim() > unknown_pred.dim():
                                unknown_pred = unknown_pred.squeeze()
                            elif unknown_pred.dim() > unknown_flags.dim():
                                unknown_flags = unknown_flags.squeeze()
                        
                        unknown_tp += ((unknown_pred == 1) & (unknown_flags == 1)).sum().item()
                        unknown_fp += ((unknown_pred == 1) & (unknown_flags == 0)).sum().item()
                        unknown_tn += ((unknown_pred == 0) & (unknown_flags == 0)).sum().item()
                        unknown_fn += ((unknown_pred == 0) & (unknown_flags == 1)).sum().item()
                else:
                    _, predicted = logits.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
        
        accuracy = 100. * correct / total if total > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'loss': total_loss / len(val_loader)
        }
        
        # Add unknown detection metrics
        if has_unknown and (unknown_tp + unknown_fp + unknown_tn + unknown_fn) > 0:
            precision = unknown_tp / (unknown_tp + unknown_fp) if (unknown_tp + unknown_fp) > 0 else 0
            recall = unknown_tp / (unknown_tp + unknown_fn) if (unknown_tp + unknown_fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics.update({
                'unknown_precision': precision,
                'unknown_recall': recall,
                'unknown_f1': f1,
                'unknown_accuracy': (unknown_tp + unknown_tn) / (unknown_tp + unknown_fp + unknown_tn + unknown_fn)
            })
        
        return total_loss / len(val_loader), metrics
    
    def _get_scheduler(self, optimizer, total_steps, warmup_steps):
        """Get learning rate scheduler with warmup"""
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1. + np.cos(np.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def _save_checkpoint(self, task_id: str, epoch: int, accuracy: float):
        """Save model checkpoint"""
        checkpoint = {
            'task_id': task_id,
            'epoch': epoch,
            'accuracy': accuracy,
            'model_state': self.model.state_dict(),
        }
        
        path = f"{self.save_dir}/{task_id}_best.pth"
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")
    
    def evaluate_all_tasks(self, test_loaders: Dict[str, torch.utils.data.DataLoader]) -> Dict:
        """Evaluate on all tasks"""
        results = {}
        
        for task_id, loader in test_loaders.items():
            # Fixed: Corrected the unpacking assignment
            _, metrics = self._validate(task_id, loader, int(task_id.split('_')[1]))
            results[task_id] = metrics['accuracy']
            
            # Update metrics tracker if available
            if self.metrics is not None:
                task_idx = int(task_id.split('_')[1])
                self.metrics.update(task_idx, task_idx, metrics['accuracy'])

            print(f"Task [{task_id}] Acc:: {metrics['accuracy']}")
        return results
    
    def get_metrics_summary(self) -> Dict:
        """Get comprehensive metrics summary"""
        if self.metrics is not None:
            return self.metrics.get_summary()
        else:
            return {}

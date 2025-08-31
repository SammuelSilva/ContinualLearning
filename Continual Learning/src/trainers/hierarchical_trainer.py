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
        
        print(f"Training {task_id} (Task {task_idx})")
        
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
            if hasattr(self.model, 'set_gradient_checkpointing'):
                self.model.set_gradient_checkpointing(True)

            train_loss, train_metrics = self._train_epoch(
                task_id, train_loader, optimizer, scheduler, task_idx, epoch
            )
            
            if hasattr(self.model, 'set_gradient_checkpointing'):
                self.model.set_gradient_checkpointing(False)
            # Validation phase
            val_loss, val_metrics = self._validate(task_id, val_loader, task_idx)
            
            # Log metrics
            print(
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
                    print(f"Early stopping at epoch {epoch+1}")
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
        """Train for one epoch with unknown sample handling and memory replay"""
        
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        unknown_tp = 0
        unknown_fp = 0
        unknown_tn = 0
        unknown_fn = 0
        replay_count = 0
        
        pbar = tqdm(train_loader, desc=f"Training epoch {epoch+1}")
        
        for batch_idx, batch_data in enumerate(pbar):
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
            
            # MEMORY REPLAY: Add samples from buffer if available
            if self.memory_buffer is not None and len(self.memory_buffer) > 0 and task_idx > 0:
                # Sample half the batch size from buffer
                replay_batch_size = min(len(images) // 2, len(self.memory_buffer))
                
                if replay_batch_size > 0:
                    replay_data = self.memory_buffer.sample(batch_size=replay_batch_size)
                    
                    if replay_data['images'].shape[0] > 0:
                        replay_images = replay_data['images'].to(self.device)
                        replay_labels = replay_data['labels'].to(self.device)
                        
                        # Mark replay samples as unknown for current task
                        replay_unknown_flags = torch.ones(
                            len(replay_images), 
                            dtype=torch.float, 
                            device=self.device
                        )
                        
                        # Combine current batch with replay samples
                        images = torch.cat([images, replay_images], dim=0)
                        labels = torch.cat([labels, replay_labels], dim=0)
                        unknown_flags = torch.cat([unknown_flags, replay_unknown_flags], dim=0)
                        has_unknown = True
                        replay_count += replay_batch_size
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images, task_id=task_id)
            
            # Handle different output formats
            if isinstance(outputs, torch.Tensor):
                logits = outputs
                outputs = {'logits': logits}
            elif isinstance(outputs, dict):
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
                        
                        # Ensure same shape
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
                'replay': replay_count,
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        # Calculate metrics
        accuracy = 100. * correct / total if total > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'loss': total_loss / len(train_loader),
            'replay_samples': replay_count
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
        
        if replay_count > 0:
            print(f"  Used {replay_count} replay samples from memory buffer")
        
        return total_loss / len(train_loader), metrics
    
    def evaluate_task(self, test_loader, task_id: str) -> float:
        """
        Replace existing evaluate_task with memory-efficient version
        """
        self.model.eval()
        
        # Disable gradient checkpointing for evaluation (faster)
        if hasattr(self.model, 'set_gradient_checkpointing'):
            original_state = self.model.gradient_checkpointing
            self.model.set_gradient_checkpointing(False)
        
        correct = 0
        total = 0
        max_batch_size = 32  # Process in smaller chunks if needed
        
        with torch.no_grad():
            for batch_data in test_loader:
                if len(batch_data) == 3:
                    images, labels, _ = batch_data
                else:
                    images, labels = batch_data
                
                batch_size = images.size(0)
                
                # Process in sub-batches to save memory
                for i in range(0, batch_size, max_batch_size):
                    end_idx = min(i + max_batch_size, batch_size)
                    sub_images = images[i:end_idx].to(self.device, non_blocking=True)
                    sub_labels = labels[i:end_idx].to(self.device, non_blocking=True)
                    
                    # Forward pass
                    outputs = self.model(sub_images, task_id=task_id)
                    
                    # Handle output format
                    if isinstance(outputs, torch.Tensor):
                        logits = outputs
                    else:
                        logits = outputs['logits']
                    
                    # Remove unknown class for accuracy calculation
                    if logits.size(1) > self.model.task_classes[task_id]:
                        logits = logits[:, :-1]
                    
                    _, predicted = logits.max(1)
                    correct += predicted.eq(sub_labels).sum().item()
                    total += sub_labels.size(0)
                    
                    # Clear intermediate tensors
                    del sub_images, sub_labels, logits
        
        # Restore gradient checkpointing state
        if hasattr(self.model, 'set_gradient_checkpointing'):
            self.model.set_gradient_checkpointing(original_state)
        
        accuracy = 100. * correct / total
        return accuracy

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
        task_idx: int,
        use_cpu: bool = True  # New parameter
    ) -> Tuple[float, Dict]:
        """
        Modified validation to optionally use CPU for memory efficiency
        """
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        unknown_tp = 0
        unknown_fp = 0
        unknown_tn = 0
        unknown_fn = 0
        
        # Determine device
        #if use_cpu:
        #    device = torch.device('cpu')
        #    # Move model to CPU temporarily
        #    original_device = next(self.model.parameters()).device
        #    self.model.cpu()
        #else:
        device = self.device
        original_device = device
        
        with torch.no_grad():
            for batch_data in val_loader:
                # Handle different batch formats
                if len(batch_data) == 3:
                    images, labels, unknown_flags = batch_data
                    images = images.to(device)
                    labels = labels.to(device)
                    unknown_flags = unknown_flags.to(device).float()
                    has_unknown = True
                else:
                    images, labels = batch_data
                    images = images.to(device)
                    labels = labels.to(device)
                    unknown_flags = torch.zeros_like(labels, dtype=torch.float, device=device)
                    has_unknown = False
                
                outputs = self.model(images, task_id=task_id)
                
                # Handle output format
                if isinstance(outputs, torch.Tensor):
                    logits = outputs
                    outputs = {'logits': logits}
                elif isinstance(outputs, dict):
                    if 'logits' not in outputs:
                        raise ValueError("Model output dictionary must contain 'logits' key")
                    logits = outputs['logits']
                else:
                    raise ValueError(f"Expected model output to be torch.Tensor or dict, got {type(outputs)}")
                
                # Compute loss on CPU
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
        
        # Move model back to original device
        if use_cpu and original_device.type == 'cuda':
            self.model.to(original_device)
        
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
        print(f"Saved checkpoint: {path}")
    
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

    def evaluate_buffer_metrics(self, task_id: str, task_idx: int):
        """
        Evaluate model performance on memory buffer samples.
        Track routing accuracy and per-task metrics.
        """
        if self.memory_buffer is None or len(self.memory_buffer) == 0:
            print("No samples in memory buffer to evaluate")
            return
        
        print(f"\n{'='*60}")
        print(f"BUFFER EVALUATION AFTER TASK {task_idx}")
        print(f"{'='*60}")
        
        # Sample from buffer
        batch_size = min(256, len(self.memory_buffer))
        memory_batch = self.memory_buffer.sample(batch_size=batch_size)
        
        images = memory_batch['images'].to(self.device)
        labels = memory_batch['labels'].to(self.device)
        task_ids_true = memory_batch.get('task_ids', [])
        
        self.model.eval()
        with torch.no_grad():
            # Get routing predictions
            predicted_tasks, confidences = self.model.predict_task_id(images)
            
            # Track routing accuracy per task
            routing_stats = {}
            task_accuracies = {}
            
            # Get unique tasks in buffer
            unique_tasks = list(set(task_ids_true))
            
            for src_task in unique_tasks:
                # Find samples from this task
                task_mask = torch.tensor([tid == src_task for tid in task_ids_true], device=self.device)
                if task_mask.sum() == 0:
                    continue
                
                task_samples = images[task_mask]
                task_labels = labels[task_mask]
                task_predicted = [predicted_tasks[i] for i, m in enumerate(task_mask) if m]
                
                # Calculate routing accuracy for this task
                correct_routing = sum(1 for pred in task_predicted if pred == src_task)
                routing_accuracy = 100.0 * correct_routing / len(task_predicted)
                
                # Track where samples were misrouted
                misrouted_to = {}
                for pred in task_predicted:
                    if pred != src_task:
                        misrouted_to[pred] = misrouted_to.get(pred, 0) + 1
                
                routing_stats[src_task] = {
                    'routing_accuracy': routing_accuracy,
                    'total_samples': len(task_predicted),
                    'correct_routing': correct_routing,
                    'misrouted_to': misrouted_to
                }
                
                # Calculate classification accuracy for correctly routed samples
                correctly_routed_mask = torch.tensor([pred == src_task for pred in task_predicted], device=self.device)
                if correctly_routed_mask.sum() > 0:
                    correct_samples = task_samples[correctly_routed_mask]
                    correct_labels = task_labels[correctly_routed_mask]
                    
                    # Get predictions for correctly routed samples
                    logits = self.model(correct_samples, task_id=src_task)
                    if isinstance(logits, dict):
                        logits = logits['logits']
                    
                    # Exclude unknown class for accuracy calculation
                    logits_known = logits[:, :-1]
                    _, predicted = logits_known.max(1)
                    
                    task_acc = 100.0 * predicted.eq(correct_labels).sum().item() / len(correct_labels)
                    task_accuracies[src_task] = task_acc
                else:
                    task_accuracies[src_task] = 0.0
            
            # Calculate overall metrics
            total_samples = len(images)
            correct_routing_total = sum(1 for i, true_task in enumerate(task_ids_true) 
                                    if predicted_tasks[i] == true_task)
            overall_routing_acc = 100.0 * correct_routing_total / total_samples
            
            # Calculate overall accuracy (only on correctly routed samples)
            correct_predictions = 0
            total_evaluated = 0
            
            for i, (true_task, pred_task) in enumerate(zip(task_ids_true, predicted_tasks)):
                if true_task == pred_task:
                    # This sample was correctly routed
                    logits = self.model(images[i:i+1], task_id=true_task)
                    if isinstance(logits, dict):
                        logits = logits['logits']
                    
                    logits_known = logits[:, :-1]
                    _, predicted = logits_known.max(1)
                    
                    if predicted.item() == labels[i].item():
                        correct_predictions += 1
                    total_evaluated += 1
            
            overall_accuracy = 100.0 * correct_predictions / total_evaluated if total_evaluated > 0 else 0.0
            
            # Print results
            print(f"\nBuffer contains {total_samples} samples from {len(unique_tasks)} tasks")
            print(f"\n📊 OVERALL METRICS:")
            print(f"  • Overall Routing Accuracy: {overall_routing_acc:.2f}%")
            print(f"  • Overall Classification Accuracy: {overall_accuracy:.2f}%")
            print(f"  • Routing Error Rate: {100.0 - overall_routing_acc:.2f}%")
            
            print(f"\n📈 PER-TASK ROUTING METRICS:")
            for task in sorted(routing_stats.keys()):
                stats = routing_stats[task]
                print(f"\n  Task {task}:")
                print(f"    • Samples: {stats['total_samples']}")
                print(f"    • Routing Accuracy: {stats['routing_accuracy']:.2f}%")
                print(f"    • Classification Accuracy: {task_accuracies.get(task, 0.0):.2f}%")
                
                if stats['misrouted_to']:
                    print(f"    • Misrouted to:")
                    for wrong_task, count in sorted(stats['misrouted_to'].items()):
                        percentage = 100.0 * count / stats['total_samples']
                        print(f"      - {wrong_task}: {count} samples ({percentage:.1f}%)")
            
            print(f"{'='*60}\n")
            
            # Store metrics for later analysis
            if not hasattr(self, 'buffer_eval_history'):
                self.buffer_eval_history = []
            
            self.buffer_eval_history.append({
                'task_idx': task_idx,
                'overall_routing_acc': overall_routing_acc,
                'overall_accuracy': overall_accuracy,
                'per_task_stats': routing_stats,
                'per_task_accuracy': task_accuracies
            })
            
            return {
                'overall_routing_accuracy': overall_routing_acc,
                'overall_accuracy': overall_accuracy,
                'routing_stats': routing_stats,
                'task_accuracies': task_accuracies
            }

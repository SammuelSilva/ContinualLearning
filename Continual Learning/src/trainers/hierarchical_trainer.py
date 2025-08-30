"""
Adapted HierarchicalTrainer to work with the new HierarchicalLoRAViT structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from src.trainers.continual_trainer import ContinualTrainer

class HierarchicalTrainer(ContinualTrainer):
    """
    Trainer adapted for new hierarchical structure with intelligent merging
    """
    
    def __init__(
        self,
        model,  # HierarchicalLoRAViT
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
        self.num_tasks = num_tasks
        
        # Store reference to memory buffer
        self.memory_buffer = memory_buffer
        
        # Track metrics
        self.block_metrics = {
            'merge_attempts': [],
            'successful_merges': [],
            'specialist_count': [],
            'block_count': []
        }
    
    def train_task(
        self,
        task_id: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 20,
        patience: int = 3,
        task_idx: Optional[int] = None,
        use_amp: bool = True  # Automatic Mixed Precision
    ):
        """
        Memory-efficient training using gradient checkpointing
        """
        print(f"\n{'='*60}")
        print(f"Training Task {task_id} (Checkpoint-Based)")
        print(f"{'='*60}\n")
        
        # Set active task
        self.model.set_active_task(task_id)
        
        # Get trainable parameters
        params = self.model.get_trainable_parameters()
        if not params:
            print(f"Warning: No trainable parameters for {task_id}")
            return 0.0
        
        # Create optimizer and scaler for mixed precision
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        scaler = GradScaler() if use_amp else None
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training with checkpointing
            train_loss, train_acc = self._train_epoch_checkpoint(
                train_loader, optimizer, task_id, scaler, use_amp
            )
            
            # Validation
            val_loss, val_acc = self._validate_checkpoint(
                val_loader, task_id, use_amp
            )
            
            scheduler.step()
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self._save_checkpoint(task_id, epoch, val_acc)
            else:
                patience_counter += 1
            
            print(f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Mark task as trained
        self.model.mark_task_trained(task_id)
        
        # Update memory buffer
        self._update_memory_buffer(train_loader, task_id)
        
        # Run OOD alignment
        if len(self.memory_buffer) > 0:
            self.align_ood_detection_checkpoint(num_epochs=3, use_amp=use_amp)
        
        # Track metrics
        stats = self.model.get_statistics()
        self.block_metrics['specialist_count'].append(stats['specialist_tasks'])
        self.block_metrics['block_count'].append(stats['num_merged_blocks'])
        self.block_metrics['merge_attempts'].append(stats['merge_attempts'])
        self.block_metrics['successful_merges'].append(stats['successful_merges'])
        
        return best_val_acc
    
    def _train_epoch_checkpoint(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        task_id: str,
        scaler: GradScaler,
        use_amp: bool
    ) -> Tuple[float, float]:
        """
        Training epoch using gradient checkpointing and mixed precision
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
                
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast(self.device.type):
                    # Forward pass with checkpointing
                    logits = self._forward_with_checkpoint(images, task_id)
                    
                    # Debug: Print shapes to understand the issue
                    print(f"Debug - Logits shape: {logits.shape}, Labels shape: {labels.shape}")
                    
                    # Classification loss - Remove unknown class logit
                    class_logits = logits[:, :-1]  # Remove last column (unknown class)
                    loss = F.cross_entropy(class_logits, labels)
                    
                    # Memory replay for preventing forgetting
                    if len(self.memory_buffer) > 0:
                        loss = self._add_memory_replay_loss(loss, task_id, batch_size=8)
                
                # Backward with mixed precision
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision
                logits = self._forward_with_checkpoint(images, task_id)
                
                # Classification loss - Remove unknown class logit
                class_logits = logits[:, :-1]  # Remove last column (unknown class)
                loss = F.cross_entropy(class_logits, labels)
                
                if len(self.memory_buffer) > 0:
                    loss = self._add_memory_replay_loss(loss, task_id, batch_size=8)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), 1.0)
                optimizer.step()
            
            # Metrics
            with torch.no_grad():
                class_logits = logits[:, :-1]  # Remove unknown class for predictions
                predictions = torch.argmax(class_logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'acc': f'{100*correct/total:.1f}%'})
                
        return total_loss / len(train_loader), 100 * correct / total
    
    def _forward_with_checkpoint(self, images: torch.Tensor, task_id: str) -> torch.Tensor:
        """Forward pass using gradient checkpointing"""
        x = self.model.backbone.patch_embed(images)
        
        # Add class token if the model expects it
        if hasattr(self.model.backbone, 'cls_token'):
            cls_token = self.model.backbone.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            
        # For timm models, manually checkpoint each block
        if hasattr(self.model.backbone, 'pos_drop'):
            x = self.model.backbone.pos_drop(x + self.model.backbone.pos_embed)
        
        for block in self.model.backbone.blocks:
            # Use model.training instead of self.training
            if self.model.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        x = self.model.backbone.norm(x)
        
        # Extract class token (first token) for classification
        if hasattr(self.model.backbone, 'cls_token'):
            # Class token is at position 0
            cls_output = x[:, 0]  # Shape: [batch_size, embed_dim]
        else:
            # If no class token, use global average pooling
            cls_output = x.mean(dim=1)  # Shape: [batch_size, embed_dim]
        
        # Apply head if backbone has one
        if hasattr(self.model.backbone, 'head') and self.model.backbone.head is not None:
            features = self.model.backbone.head(cls_output)
        else:
            features = cls_output
        
        # Get appropriate task head
        logits = None
        if task_id in self.model.task_heads:
            logits = self.model.task_heads[task_id](features)
        else:
            for block in self.model.merged_blocks:
                if task_id in block.task_ids:
                    logits = block.task_heads[task_id](features)
                    break
        
        if logits is None:
            raise ValueError(f"No head found for task_id: {task_id}")
        
        return logits

    def _validate_checkpoint(self, val_loader: DataLoader, task_id: str, use_amp: bool) -> Tuple[float, float]:
        """
        Validation with optional mixed precision
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                if use_amp:
                    with autocast(self.device.type):
                        logits = self.model(images, task_id=task_id)
                        class_logits = logits[:, :-1]  # Remove unknown class
                        loss = F.cross_entropy(class_logits, labels)
                else:
                    logits = self.model(images, task_id=task_id)
                    class_logits = logits[:, :-1]  # Remove unknown class
                    loss = F.cross_entropy(class_logits, labels)
                
                predictions = torch.argmax(class_logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()
        
        return total_loss / len(val_loader), 100 * correct / total

    def _add_memory_replay_loss(self, loss: torch.Tensor, task_id: str, batch_size: int = 32) -> torch.Tensor:
        """
        Add memory replay loss for preventing forgetting
        """
        memory_batch = self.memory_buffer.sample(batch_size=batch_size)
        if len(memory_batch['images']) == 0:
            return loss
        
        mem_images = memory_batch['images'].to(self.device)
        mem_labels = memory_batch['labels'].to(self.device)
        mem_task_ids = memory_batch.get('task_ids', [])
        
        # Forward with checkpointing
        with autocast(self.device.type):
            mem_logits = self._forward_with_checkpoint(mem_images, task_id)
        
        # Create appropriate labels
        adjusted_labels = []
        for label, mem_task_id in zip(mem_labels, mem_task_ids):
            if mem_task_id == task_id:
                adjusted_labels.append(label.item())
            else:
                # Use the unknown class index (last class)
                adjusted_labels.append(mem_logits.shape[1] - 1)  # Unknown class
        
        adjusted_labels = torch.tensor(adjusted_labels, device=self.device)
        
        # Use full logits (including unknown class) for memory replay
        memory_loss = F.cross_entropy(mem_logits, adjusted_labels)
        
        # Combine losses
        return loss + 0.3 * memory_loss

    def align_ood_detection_checkpoint(self, num_epochs=3, use_amp=True):
        """
        OOD alignment using gradient checkpointing
        """
        if len(self.memory_buffer) == 0:
            return
        
        print(f"\n=== OOD Detection Alignment (Checkpoint-Based) ===")
        
        # Collect all heads
        all_heads = []
        for task_id in self.model.specialist_tasks:
            if task_id in self.model.task_heads:
                all_heads.append((task_id, self.model.task_heads[task_id]))
        
        for block in self.model.merged_blocks:
            for task_id in block.task_ids:
                all_heads.append((task_id, block.task_heads[task_id]))
        
        if not all_heads:
            return
        
        # Optimizer for heads
        head_params = []
        for _, head in all_heads:
            head_params.extend(head.parameters())
        
        optimizer = torch.optim.Adam(head_params, lr=1e-4)
        scaler = GradScaler() if use_amp else None
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            
            # Sample from memory
            batch = self.memory_buffer.sample(batch_size=32)
            if len(batch['images']) == 0:
                continue
            
            images = batch['images'].to(self.device)
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast(self.device.type):
                    # Get features with checkpointing
                    features = checkpoint(self.model.backbone, images, use_reentrant=False)
                    
                    total_loss = 0
                    for task_id, head in all_heads:
                        logits = head(features)
                        unknown_labels = torch.full(
                            (len(features),),
                            logits.shape[1] - 1,
                            device=self.device
                        )
                        loss = F.cross_entropy(logits, unknown_labels)
                        total_loss += loss
                    
                    avg_loss = total_loss / len(all_heads)
                
                scaler.scale(avg_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                features = checkpoint(self.model.backbone, images, use_reentrant=False)
                
                total_loss = 0
                for task_id, head in all_heads:
                    logits = head(features)
                    unknown_labels = torch.full(
                        (len(features),),
                        logits.shape[1] - 1,
                        device=self.device
                    )
                    loss = F.cross_entropy(logits, unknown_labels)
                    total_loss += loss
                
                avg_loss = total_loss / len(all_heads)
                avg_loss.backward()
                optimizer.step()
            
            epoch_loss = avg_loss.item()
            print(f"  Epoch {epoch+1}/{num_epochs}: Loss = {epoch_loss:.4f}")

    def _update_memory_buffer(self, train_loader: DataLoader, task_id: str):
        """
        Update memory buffer with task data
        """
        print("Updating memory buffer...")
        
        all_images = []
        all_labels = []
        all_features = []
        
        self.model.eval()
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(self.device)
                features = self.model(images, task_id=task_id, return_features=True)
                
                all_images.append(images.cpu())
                all_labels.append(labels)
                all_features.append(features.cpu())
                
                if len(all_images) * images.shape[0] >= 100:  # Limit samples
                    break
        
        if all_images:
            all_images = torch.cat(all_images, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_features = torch.cat(all_features, dim=0)
            
            self.memory_buffer.update(all_images, all_labels, task_id, all_features)
        
        print(f"Buffer updated. Current size: {len(self.memory_buffer)}")
    
    def _save_checkpoint(self, task_id: str, epoch: int, val_acc: float):
        """
        Save checkpoint
        """
        checkpoint_path = os.path.join(
            self.save_dir,
            f"{task_id}_epoch{epoch}_acc{val_acc:.2f}.pt"
        )
        self.model.save_checkpoint(checkpoint_path)
    
    def get_metrics_summary(self) -> Dict:
        """
        Get training metrics summary
        """
        stats = self.model.get_statistics()
        return {
            'total_tasks': stats['total_tasks'],
            'specialist_tasks': stats['specialist_tasks'],
            'merged_blocks': stats['num_merged_blocks'],
            'merge_attempts': stats['merge_attempts'],
            'successful_merges': stats['successful_merges'],
            'merge_success_rate': (stats['successful_merges'] / max(1, stats['merge_attempts'])) * 100,
            'block_metrics': self.block_metrics
        }

    def evaluate_all_tasks(self, test_loaders: Dict[str, DataLoader]) -> Dict:
        """
        Evaluate on all tasks using hierarchical prediction
        """
        self.model.eval()
        results = {
            'task_accuracy': {},
            'confusion_matrix': np.zeros((len(test_loaders), len(test_loaders)))
        }
        
        for true_task_id, test_loader in test_loaders.items():
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Predict task
                    predicted_tasks, confidences = self.model.predict_task_id(images)
                    
                    # Evaluate predictions
                    for i, (pred_task, label) in enumerate(zip(predicted_tasks, labels)):
                        if pred_task == true_task_id:
                            # Get class prediction
                            logits = self.model(images[i:i+1], task_id=pred_task)
                            pred_class = torch.argmax(logits[:, :-1], dim=1)
                            if pred_class == label:
                                correct += 1
                        
                        total += 1
                        
                        # Update confusion matrix
                        true_idx = int(true_task_id.split('_')[1])
                        pred_idx = int(pred_task.split('_')[1])
                        results['confusion_matrix'][true_idx, pred_idx] += 1
            
            accuracy = 100 * correct / total if total > 0 else 0
            results['task_accuracy'][true_task_id] = accuracy
            print(f"Task {true_task_id}: {accuracy:.2f}%")
        
        if hasattr(self.model, "visualize_hierarchy"):
            self.model.visualize_hierarchy()

        results['avg_accuracy'] = np.mean(list(results['task_accuracy'].values()))
        return results

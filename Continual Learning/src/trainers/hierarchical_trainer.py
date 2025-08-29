"""
Adapted HierarchicalTrainer to work with the new HierarchicalLoRAViT structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
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
        task_idx: Optional[int] = None
    ):
        """
        Memory-efficient training using cached features on CPU
        """
        print(f"\n{'='*60}")
        print(f"Training Task {task_id} (Memory-Efficient Mode)")
        print(f"{'='*60}\n")
        
        # Set active task
        self.model.set_active_task(task_id)
        
        # Get trainable parameters
        params = self.model.get_trainable_parameters()
        if not params:
            print(f"Warning: No trainable parameters for {task_id} (might be in frozen block)")
            return 0.0
        
        # Step 1: Pre-compute all features on CPU
        print("Pre-computing features on CPU...")
        train_features, train_labels = self._extract_features_cpu(train_loader)
        val_features, val_labels = self._extract_features_cpu(val_loader)
        
        # Create feature dataloaders
        train_dataset = TensorDataset(train_features, train_labels)
        feature_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        
        val_dataset = TensorDataset(val_features, val_labels)
        val_feature_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        
        # Move backbone back to GPU for memory operations
        self.model.backbone.to(self.device)
        
        # Create optimizer (only for lightweight heads/adapters)
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training with cached features
            train_loss, train_acc = self._train_epoch_cached(
                feature_loader, optimizer, task_id, params
            )
            
            # Validation with cached features
            val_loss, val_acc = self._validate_cached(
                val_feature_loader, task_id
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
        
        # Run OOD alignment if we have memory
        if len(self.memory_buffer) > 0:
            self.align_ood_detection_efficient(num_epochs=3)
        
        # Track metrics
        stats = self.model.get_statistics()
        self.block_metrics['specialist_count'].append(stats['specialist_tasks'])
        self.block_metrics['block_count'].append(stats['num_merged_blocks'])
        self.block_metrics['merge_attempts'].append(stats['merge_attempts'])
        self.block_metrics['successful_merges'].append(stats['successful_merges'])
        
        return best_val_acc
    
    def _extract_features_cpu(self, data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features on CPU to save GPU memory
        """
        self.model.backbone.cpu()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc="Extracting features"):
                features = self.model.backbone(images)
                all_features.append(features)
                all_labels.append(labels)
        
        return torch.cat(all_features), torch.cat(all_labels)
    
    def _train_epoch_cached(
        self,
        feature_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        task_id: str,
        params
    ) -> Tuple[float, float]:
        """
        Train epoch using pre-cached features (GPU efficient)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Get the appropriate head
        if task_id in self.model.task_heads:
            head = self.model.task_heads[task_id]
        else:
            # Task is in a merged block
            for block in self.model.merged_blocks:
                if task_id in block.task_ids:
                    head = block.task_heads[task_id]
                    break
        
        pbar = tqdm(feature_loader, desc="Training")
        for features, labels in pbar:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward through task head only (very fast)
            logits = head(features)
            
            # Task loss
            loss = F.cross_entropy(logits[:, :-1], labels)
            
            # OOD loss with memory samples
            if len(self.memory_buffer) > 0:
                memory_batch = self.memory_buffer.sample(batch_size=16)
                if len(memory_batch['images']) > 0:
                    # Need backbone for memory samples
                    with torch.no_grad():
                        mem_features = self.model.backbone(memory_batch['images'].to(self.device))
                    
                    mem_logits = head(mem_features)
                    unknown_labels = torch.full(
                        (len(mem_features),),
                        logits.shape[1] - 1,
                        dtype=torch.long,
                        device=self.device
                    )
                    loss_unknown = F.cross_entropy(mem_logits, unknown_labels)
                    loss = loss + self.lambda_unknown * loss_unknown
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            # Metrics
            with torch.no_grad():
                predictions = torch.argmax(logits[:, :-1], dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'acc': f'{100*correct/total:.1f}%'})
        
        return total_loss / len(feature_loader), 100 * correct / total
    
    def _validate_cached(
        self,
        val_feature_loader: DataLoader,
        task_id: str
    ) -> Tuple[float, float]:
        """
        Validate using pre-cached features
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # Get the appropriate head
        if task_id in self.model.task_heads:
            head = self.model.task_heads[task_id]
        else:
            for block in self.model.merged_blocks:
                if task_id in block.task_ids:
                    head = block.task_heads[task_id]
                    break
        
        with torch.no_grad():
            for features, labels in val_feature_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                logits = head(features)
                loss = F.cross_entropy(logits[:, :-1], labels)
                
                predictions = torch.argmax(logits[:, :-1], dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()
        
        return total_loss / len(val_feature_loader), 100 * correct / total
    
    def align_ood_detection_efficient(self, num_epochs=3, max_samples=1000):
        """
        Memory-efficient OOD alignment
        """
        if len(self.memory_buffer) == 0:
            return
        
        print(f"\n=== OOD Detection Alignment (Efficient) ===")
        
        # Get subset of memory
        all_data = self.memory_buffer.get_all_data()
        if len(all_data['images']) > max_samples:
            indices = torch.randperm(len(all_data['images']))[:max_samples]
            subset_images = all_data['images'][indices]
            subset_labels = all_data['labels'][indices]
        else:
            subset_images = all_data['images']
            subset_labels = all_data['labels']
        
        # Pre-compute features on CPU
        self.model.backbone.cpu()
        with torch.no_grad():
            features = []
            for i in range(0, len(subset_images), 32):
                batch = subset_images[i:i+32]
                features.append(self.model.backbone(batch))
            features = torch.cat(features)
        
        # Move backbone back to GPU
        self.model.backbone.to(self.device)
        
        # Create dataset
        dataset = TensorDataset(features, subset_labels)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Get all heads
        all_heads = []
        for task_id in self.model.specialist_tasks:
            if task_id in self.model.task_heads:
                all_heads.append(self.model.task_heads[task_id])
        
        for block in self.model.merged_blocks:
            for task_id in block.task_ids:
                all_heads.append(block.task_heads[task_id])
        
        # Optimize
        head_params = []
        for head in all_heads:
            head_params.extend(head.parameters())
        
        optimizer = torch.optim.Adam(head_params, lr=1e-4)
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch_features, _ in dataloader:
                batch_features = batch_features.to(self.device)
                
                total_loss = 0
                for head in all_heads:
                    logits = head(batch_features)
                    unknown_labels = torch.full(
                        (len(batch_features),),
                        logits.shape[1] - 1,
                        device=self.device
                    )
                    loss = F.cross_entropy(logits, unknown_labels)
                    total_loss += loss
                
                avg_loss = total_loss / len(all_heads)
                optimizer.zero_grad()
                avg_loss.backward()
                optimizer.step()
                
                epoch_loss += avg_loss.item()
            
            print(f"  Epoch {epoch+1}/{num_epochs}: Loss = {epoch_loss/len(dataloader):.4f}")
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        task_id: str
    ) -> Tuple[float, float]:
        """
        Training epoch with unknown class loss
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            logits = self.model(images, task_id=task_id)
            
            # Classification loss (excluding unknown class)
            loss = F.cross_entropy(logits[:, :-1], labels)
            
            # Add unknown class loss if we have memory
            if len(self.memory_buffer) > 0:
                memory_batch = self.memory_buffer.sample(batch_size=min(16, len(images)))
                if len(memory_batch['images']) > 0:
                    mem_images = memory_batch['images'].to(self.device)
                    mem_logits = self.model(mem_images, task_id=task_id)
                    
                    # Label memory samples as unknown
                    unknown_labels = torch.full(
                        (len(mem_images),),
                        logits.shape[1] - 1,  # Unknown class index
                        dtype=torch.long,
                        device=self.device
                    )
                    loss_unknown = F.cross_entropy(mem_logits, unknown_labels)
                    loss = loss + self.lambda_unknown * loss_unknown
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), 1.0)
            optimizer.step()
            
            # Metrics
            with torch.no_grad():
                predictions = torch.argmax(logits[:, :-1], dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'acc': f'{100*correct/total:.1f}%'})
        
        return total_loss / len(train_loader), 100 * correct / total
    
    def _validate(self, val_loader: DataLoader, task_id: str) -> Tuple[float, float]:
        """
        Validate on validation set
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(images, task_id=task_id)
                loss = F.cross_entropy(logits[:, :-1], labels)
                
                predictions = torch.argmax(logits[:, :-1], dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()
        
        return total_loss / len(val_loader), 100 * correct / total
    
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
    
    def align_ood_detection(self, num_epochs=3):
        """
        Simplified OOD alignment for unknown class detection
        """
        if len(self.memory_buffer) == 0:
            return
        
        print(f"\n=== OOD Detection Alignment ===")
        
        # Get all task heads that need alignment
        all_heads = []
        
        # Specialist task heads
        for task_id in self.model.specialist_tasks:
            if task_id in self.model.task_heads:
                all_heads.append((task_id, self.model.task_heads[task_id], 'specialist'))
        
        # Merged block heads
        for block in self.model.merged_blocks:
            for task_id in block.task_ids:
                all_heads.append((task_id, block.task_heads[task_id], 'block'))
        
        if not all_heads:
            return
        
        # Create optimizer for all heads
        head_params = []
        for _, head, _ in all_heads:
            head_params.extend(head.parameters())
        
        optimizer = torch.optim.Adam(head_params, lr=1e-4)
        
        # Training loop
        for epoch in range(num_epochs):
            # Sample from memory
            batch = self.memory_buffer.sample(batch_size=32)
            if len(batch['images']) == 0:
                continue
            
            images = batch['images'].to(self.device)
            
            # Get features
            with torch.no_grad():
                features = self.model.backbone(images)
            
            total_loss = 0
            for task_id, head, head_type in all_heads:
                logits = head(features)
                
                # All memory samples are "unknown" to each task
                unknown_labels = torch.full(
                    (len(features),),
                    logits.shape[1] - 1,
                    device=self.device
                )
                
                loss = F.cross_entropy(logits, unknown_labels)
                total_loss += loss
            
            # Optimize
            optimizer.zero_grad()
            (total_loss / len(all_heads)).backward()
            optimizer.step()
            
            print(f"  Epoch {epoch+1}/{num_epochs}: Loss = {total_loss.item()/len(all_heads):.4f}")
    
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
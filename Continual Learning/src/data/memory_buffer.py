"""
Memory buffer for replay-based continual learning
"""

import random
import numpy as np
import torch

from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class MemoryBuffer:
    """
    Experience replay buffer for continual learning.
    Stores exemplars from previous tasks.
    """
    
    def __init__(
        self,
        buffer_size: int = 2000,
        selection_strategy: str = "random",  # "random", "herding", "uncertainty"
        samples_per_class: int = 20
    ):
        self.buffer_size = buffer_size
        self.selection_strategy = selection_strategy
        self.samples_per_class = samples_per_class
        
        # Storage
        self.images = []
        self.labels = []
        self.task_ids = []
        self.features = []  # Optional: store features for efficient herding
        
        # Class management
        self.class_indices = defaultdict(list)  # class_id -> buffer indices
        self.task_classes = defaultdict(list)   # task_id -> list of classes
        
    def update(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        task_id: str,
        features: Optional[torch.Tensor] = None
    ):
        """
        Update buffer with new task data.
        Implements different selection strategies.
        """
        
        # Convert to numpy for storage efficiency
        images_np = images.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        if self.selection_strategy == "random":
            indices = self._random_selection(labels_np)
        elif self.selection_strategy == "herding":
            if features is None:
                raise ValueError("Features required for herding selection")
            indices = self._herding_selection(features, labels_np)
        else:
            indices = self._random_selection(labels_np)
        
        # Add selected samples to buffer
        for idx in indices:
            self._add_sample(
                images_np[idx],
                labels_np[idx],
                task_id,
                features[idx] if features is not None else None
            )
        
        # Maintain buffer size constraint
        self._reduce_buffer()
    
    def _add_sample(
        self,
        image: np.ndarray,
        label: int,
        task_id: str,
        feature: Optional[np.ndarray] = None
    ):
        """Add a single sample to the buffer"""
        self.images.append(image)
        self.labels.append(label)
        self.task_ids.append(task_id)
        
        if feature is not None:
            self.features.append(feature)
        
        # Update indices
        buffer_idx = len(self.images) - 1
        self.class_indices[label].append(buffer_idx)
        
        if label not in self.task_classes[task_id]:
            self.task_classes[task_id].append(label)
    
    def _random_selection(self, labels: np.ndarray) -> List[int]:
        """Randomly select samples per class"""
        selected_indices = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            n_select = min(self.samples_per_class, len(label_indices))
            selected = np.random.choice(label_indices, n_select, replace=False)
            selected_indices.extend(selected.tolist())
        
        return selected_indices
    
    def _herding_selection(
        self,
        features: torch.Tensor,
        labels: np.ndarray
    ) -> List[int]:
        """
        Herding selection: choose samples closest to class mean.
        Based on iCaRL paper.
        """
        features_np = features.cpu().numpy()
        selected_indices = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            class_features = features_np[label_indices]
            
            # Compute class mean
            class_mean = np.mean(class_features, axis=0, keepdims=True)
            
            # Select samples closest to mean
            distances = np.linalg.norm(class_features - class_mean, axis=1)
            n_select = min(self.samples_per_class, len(label_indices))
            closest_indices = np.argsort(distances)[:n_select]
            
            selected_indices.extend(label_indices[closest_indices].tolist())
        
        return selected_indices
    
    def _reduce_buffer(self):
        """Reduce buffer size if it exceeds the limit"""
        if len(self.images) <= self.buffer_size:
            return
        
        # Calculate samples to keep per class
        n_classes = len(self.class_indices)
        if n_classes == 0:
            return
        
        samples_per_class = self.buffer_size // n_classes
        
        # Select samples to keep
        keep_indices = []
        for class_id, indices in self.class_indices.items():
            if len(indices) <= samples_per_class:
                keep_indices.extend(indices)
            else:
                # Randomly select samples to keep
                keep = random.sample(indices, samples_per_class)
                keep_indices.extend(keep)
        
        # Sort indices for efficient removal
        keep_indices = sorted(keep_indices)
        
        # Create new buffer with selected samples
        self.images = [self.images[i] for i in keep_indices]
        self.labels = [self.labels[i] for i in keep_indices]
        self.task_ids = [self.task_ids[i] for i in keep_indices]
        
        if self.features:
            self.features = [self.features[i] for i in keep_indices]
        
        # Rebuild indices
        self._rebuild_indices()
    
    def _rebuild_indices(self):
        """Rebuild class indices after buffer reduction"""
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_indices[label].append(idx)
    
    def sample(
        self,
        batch_size: int,
        exclude_task: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a batch from the buffer.
        Can exclude samples from a specific task.
        """
        if len(self.images) == 0:
            return {
                'images': torch.tensor([]),
                'labels': torch.tensor([]),
                'task_ids': []
            }
        
        # Get valid indices
        valid_indices = list(range(len(self.images)))
        
        if exclude_task is not None:
            valid_indices = [
                i for i in valid_indices 
                if self.task_ids[i] != exclude_task
            ]
        
        if len(valid_indices) == 0:
            return {
                'images': torch.tensor([]),
                'labels': torch.tensor([]),
                'task_ids': []
            }
        
        # Sample indices
        sample_size = min(batch_size, len(valid_indices))
        sampled_indices = random.sample(valid_indices, sample_size)
        
        # Gather samples
        images = torch.tensor(
            np.array([self.images[i] for i in sampled_indices]),
            dtype=torch.float32
        )
        labels = torch.tensor(
            [self.labels[i] for i in sampled_indices],
            dtype=torch.long
        )
        task_ids = [self.task_ids[i] for i in sampled_indices]
        
        return {
            'images': images,
            'labels': labels,
            'task_ids': task_ids
        }
    
    def get_all_data(self) -> Dict[str, torch.Tensor]:
        """Get all data from buffer"""
        if len(self.images) == 0:
            return {
                'images': torch.tensor([]),
                'labels': torch.tensor([]),
                'task_ids': []
            }
        
        return {
            'images': torch.tensor(np.array(self.images), dtype=torch.float32),
            'labels': torch.tensor(self.labels, dtype=torch.long),
            'task_ids': self.task_ids
        }
    
    def __len__(self) -> int:
        return len(self.images)
    
    def get_statistics(self) -> Dict:
        """Get buffer statistics"""
        stats = {
            'total_samples': len(self.images),
            'num_classes': len(self.class_indices),
            'num_tasks': len(self.task_classes),
            'samples_per_class': {},
            'samples_per_task': {}
        }
        
        # Samples per class
        for class_id, indices in self.class_indices.items():
            stats['samples_per_class'][class_id] = len(indices)
        
        # Samples per task
        for task_id in self.task_classes.keys():
            count = sum(1 for t in self.task_ids if t == task_id)
            stats['samples_per_task'][task_id] = count
        
        return stats
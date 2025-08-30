"""
Enhanced CIFAR-100 dataset with SVHN as unknown data for continual learning
"""

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import datasets, transforms
import numpy as np
from typing import List, Tuple, Dict, Optional
import random


class SVHNUnknownDataset(Dataset):
    """
    SVHN dataset wrapper to be used as unknown/out-of-distribution data
    """
    
    def __init__(
        self,
        data_root: str = "./data",
        split: str = "train",
        transform: Optional[transforms.Compose] = None
    ):
        self.data_root = data_root
        
        # Use SVHN as unknown data
        if split == "train":
            self.dataset = datasets.SVHN(
                root=data_root,
                split='train',
                download=True,
                transform=transform
            )
        else:
            self.dataset = datasets.SVHN(
                root=data_root,
                split='test',
                download=True,
                transform=transform
            )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, _ = self.dataset[idx]  # Ignore original label
        # Return -1 as label to indicate unknown/OOD sample
        return image, -1


class MixedTaskDataset(Dataset):
    """
    Dataset that mixes task-specific data with unknown samples
    """
    
    def __init__(
        self,
        task_dataset: Dataset,
        unknown_dataset: Dataset,
        unknown_ratio: float = 0.2,
        num_classes_per_task: int = 10
    ):
        self.task_dataset = task_dataset
        self.unknown_dataset = unknown_dataset
        self.unknown_ratio = unknown_ratio
        self.num_classes_per_task = num_classes_per_task
        
        # Calculate dataset sizes
        self.task_size = len(task_dataset)
        self.unknown_size = int(self.task_size * unknown_ratio / (1 - unknown_ratio))
        
        # Sample indices from unknown dataset
        self.unknown_indices = np.random.choice(
            len(unknown_dataset),
            size=min(self.unknown_size, len(unknown_dataset)),
            replace=False
        )
        
        self.total_size = self.task_size + len(self.unknown_indices)
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        if idx < self.task_size:
            # Get task sample
            image, label = self.task_dataset[idx]
            # Label stays as is (0 to num_classes_per_task-1)
            return image, label, 0  # 0 indicates known sample
        else:
            # Get unknown sample
            unknown_idx = self.unknown_indices[idx - self.task_size]
            image, _ = self.unknown_dataset[unknown_idx]
            # Use num_classes_per_task as the unknown class label
            return image, self.num_classes_per_task, 1  # 1 indicates unknown sample


class EnhancedContinualCIFAR100:
    """
    Enhanced CIFAR-100 dataset with SVHN unknown data for continual learning.
    """
    
    def __init__(
        self,
        data_root: str = "./data",
        num_tasks: int = 10,
        classes_per_task: int = 10,
        validation_split: float = 0.1,
        unknown_ratio: float = 0.2,
        unknown_ratio_decay: float = 0.9,  # Decay factor for unknown ratio across tasks
        use_unknown_data: bool = True,
        seed: int = 42
    ):
        self.data_root = data_root
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task
        self.validation_split = validation_split
        self.unknown_ratio = unknown_ratio
        self.unknown_ratio_decay = unknown_ratio_decay
        self.use_unknown_data = use_unknown_data
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Data augmentation for training
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])
        
        # No augmentation for validation/test
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])
        
        # Transform for SVHN (already 32x32 RGB)
        self.svhn_train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4377, 0.4438, 0.4728],  # SVHN statistics
                std=[0.1980, 0.2010, 0.1970]
            )
        ])
        
        self.svhn_test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4377, 0.4438, 0.4728],
                std=[0.1980, 0.2010, 0.1970]
            )
        ])
        
        # Load CIFAR-100
        self.train_dataset = datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        self.test_dataset = datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=self.test_transform
        )
        
        # Load SVHN as unknown data if enabled
        if self.use_unknown_data:
            self.unknown_train_dataset = SVHNUnknownDataset(
                data_root=data_root,
                split="train",
                transform=self.svhn_train_transform
            )
            
            self.unknown_test_dataset = SVHNUnknownDataset(
                data_root=data_root,
                split="test",
                transform=self.svhn_test_transform
            )
        
        # Create task splits
        self.task_classes = self._create_task_splits()
        self.current_task = 0
        
    def _create_task_splits(self) -> List[List[int]]:
        """Split 100 classes into tasks"""
        all_classes = list(range(100))
        np.random.shuffle(all_classes)
        
        task_classes = []
        for i in range(self.num_tasks):
            start_idx = i * self.classes_per_task
            end_idx = start_idx + self.classes_per_task
            task_classes.append(all_classes[start_idx:end_idx])
        
        return task_classes
    
    def get_task_unknown_ratio(self, task_id: int) -> float:
        """
        Calculate unknown ratio for a specific task.
        Higher ratio for early tasks, decay for later tasks.
        """
        if not self.use_unknown_data:
            return 0.0
        
        # For task 0, use full unknown ratio
        # For later tasks, decay the ratio
        if task_id == 0:
            return self.unknown_ratio
        else:
            # Decay ratio but maintain minimum of 0.05
            decayed_ratio = self.unknown_ratio * (self.unknown_ratio_decay ** task_id)
            return max(decayed_ratio, 0.05)
    
    def get_task_dataset(
        self,
        task_id: int,
        split: str = "train",
        include_unknown: bool = True
    ) -> Dataset:
        """Get dataset for a specific task with optional unknown samples"""
        
        if task_id >= self.num_tasks:
            raise ValueError(f"Task {task_id} out of range")
        
        classes = self.task_classes[task_id]
        
        # Get base task dataset
        if split == "train":
            task_dataset = TaskDataset(
                self.train_dataset,
                classes,
                remap_labels=True
            )
            unknown_dataset = self.unknown_train_dataset if self.use_unknown_data else None
        else:
            task_dataset = TaskDataset(
                self.test_dataset,
                classes,
                remap_labels=True
            )
            unknown_dataset = self.unknown_test_dataset if self.use_unknown_data else None
        
        # Mix with unknown data if requested
        if include_unknown and self.use_unknown_data and unknown_dataset is not None:
            print("Including unknown data with ratio:", self.get_task_unknown_ratio(task_id))
            unknown_ratio = self.get_task_unknown_ratio(task_id)
            return MixedTaskDataset(
                task_dataset=task_dataset,
                unknown_dataset=unknown_dataset,
                unknown_ratio=unknown_ratio,
                num_classes_per_task=self.classes_per_task
            )
        else:
            return task_dataset
    
    def get_task_loaders(
        self,
        task_id: int,
        batch_size: int = 128,
        num_workers: int = 4,
        include_unknown_train: bool = True,
        include_unknown_test: bool = False
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get train, val, test loaders for a task.
        
        Args:
            task_id: Task index
            batch_size: Batch size for loaders
            num_workers: Number of data loading workers
            include_unknown_train: Include unknown samples in training
            include_unknown_test: Include unknown samples in test (for evaluation)
        """
        
        # Get task datasets
        task_train = self.get_task_dataset(
            task_id, 
            "train", 
            include_unknown=include_unknown_train
        )
        task_test = self.get_task_dataset(
            task_id, 
            "test", 
            include_unknown=include_unknown_test
        )
        
        # Split train into train/val
        n_train = len(task_train)
        n_val = int(n_train * self.validation_split)
        n_train = n_train - n_val
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            task_train, [n_train, n_val]
        )
        
        # Custom collate function to handle mixed dataset output
        def collate_fn(batch):
            if isinstance(batch[0], tuple) and len(batch[0]) == 3:
                # Mixed dataset with unknown indicator
                images, labels, unknown_flags = zip(*batch)
                return (
                    torch.stack(images),
                    torch.tensor(labels, dtype=torch.long),
                    torch.tensor(unknown_flags, dtype=torch.float)
                )
            else:
                # Standard dataset
                images, labels = zip(*batch)
                return torch.stack(images), torch.tensor(labels, dtype=torch.long)
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_fn if include_unknown_train and self.use_unknown_data else None
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_fn if include_unknown_train and self.use_unknown_data else None
        )
        
        test_loader = DataLoader(
            task_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_fn if include_unknown_test and self.use_unknown_data else None
        )
        
        return train_loader, val_loader, test_loader
    
    def get_statistics(self, task_id: int) -> Dict:
        """Get statistics about the dataset composition for a task"""
        unknown_ratio = self.get_task_unknown_ratio(task_id)
        
        stats = {
            'task_id': task_id,
            'num_classes': self.classes_per_task,
            'task_classes': self.task_classes[task_id],
            'unknown_ratio': unknown_ratio,
            'unknown_enabled': self.use_unknown_data
        }
        
        if self.use_unknown_data:
            task_train = self.get_task_dataset(task_id, "train", include_unknown=False)
            stats['task_samples'] = len(task_train)
            stats['expected_unknown_samples'] = int(len(task_train) * unknown_ratio / (1 - unknown_ratio))
        
        return stats


class TaskDataset(Dataset):
    """
    Wrapper dataset for a specific task.
    Filters and remaps labels for task-specific training.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        task_classes: List[int],
        remap_labels: bool = True
    ):
        self.base_dataset = base_dataset
        self.task_classes = task_classes
        self.remap_labels = remap_labels
        
        # Create label mapping
        if remap_labels:
            self.label_map = {
                orig_class: new_label 
                for new_label, orig_class in enumerate(task_classes)
            }
        
        # Filter indices for this task
        self.indices = []
        if hasattr(base_dataset, 'targets'):
            for idx, label in enumerate(base_dataset.targets):
                if label in task_classes:
                    self.indices.append(idx)
        else:
            for idx in range(len(base_dataset)):
                _, label = base_dataset[idx]
                if label in task_classes:
                    self.indices.append(idx)
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        base_idx = self.indices[idx]
        image, label = self.base_dataset[base_idx]
        
        if self.remap_labels:
            label = self.label_map[label]
        
        return image, label
"""
CIFAR-100 dataset setup for continual learning
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from typing import List, Tuple, Dict, Optional


class ContinualCIFAR100:
    """
    CIFAR-100 dataset split into multiple tasks for continual learning.
    """
    
    def __init__(
        self,
        data_root: str = "./data",
        num_tasks: int = 10,
        classes_per_task: int = 10,
        validation_split: float = 0.1,
        seed: int = 42
    ):
        self.data_root = data_root
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task
        self.validation_split = validation_split
        
        np.random.seed(seed)
        
        # Data augmentation for training
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to ViT input size
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])
        
        # No augmentation for validation/test
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to ViT input size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
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
    
    def get_task_dataset(
        self,
        task_id: int,
        split: str = "train"
    ) -> Dataset:
        """Get dataset for a specific task"""
        
        if task_id >= self.num_tasks:
            raise ValueError(f"Task {task_id} out of range")
        
        classes = self.task_classes[task_id]
        
        if split == "train":
            return TaskDataset(
                self.train_dataset,
                classes,
                remap_labels=True
            )
        else:
            return TaskDataset(
                self.test_dataset,
                classes,
                remap_labels=True
            )
    
    def get_task_loaders(
        self,
        task_id: int,
        batch_size: int = 128,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train, val, test loaders for a task"""
        
        # Get task dataset
        task_train = self.get_task_dataset(task_id, "train")
        task_test = self.get_task_dataset(task_id, "test")
        
        # Split train into train/val
        n_train = len(task_train)
        n_val = int(n_train * self.validation_split)
        n_train = n_train - n_val
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            task_train, [n_train, n_val]
        )
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        )
        
        test_loader = DataLoader(
            task_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        )
        
        return train_loader, val_loader, test_loader
    
    def get_all_test_data(self) -> DataLoader:
        """Get test data for all tasks seen so far"""
        all_classes = []
        for i in range(self.current_task + 1):
            all_classes.extend(self.task_classes[i])
        
        dataset = TaskDataset(
            self.test_dataset,
            all_classes,
            remap_labels=False  # Keep original labels
        )
        
        return DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )


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

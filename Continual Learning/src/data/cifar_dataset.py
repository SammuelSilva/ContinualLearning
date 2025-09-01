"""
Enhanced CIFAR-100 dataset with SVHN as unknown data for continual learning
Memory-efficient version that keeps data on CPU until needed
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from typing import List, Tuple, Dict, Optional
import gc


class SVHNUnknownDataset(Dataset):
    """
    SVHN dataset wrapper to be used as unknown/out-of-distribution data
    Lazy loading version - doesn't preload data
    """
    
    def __init__(
        self,
        data_root: str = "./data",
        split: str = "train",
        transform: Optional[transforms.Compose] = None
    ):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self._dataset = None  # Lazy loading
    
    def _load_if_needed(self):
        """Load dataset only when needed"""
        if self._dataset is None:
            if self.split == "train":
                self._dataset = datasets.SVHN(
                    root=self.data_root,
                    split='train',
                    download=True,
                    transform=self.transform
                )
            else:
                self._dataset = datasets.SVHN(
                    root=self.data_root,
                    split='test',
                    download=True,
                    transform=self.transform
                )
    
    def __len__(self):
        self._load_if_needed()
        return len(self._dataset)
    
    def __getitem__(self, idx):
        self._load_if_needed()
        image, _ = self._dataset[idx]  # Ignore original label
        # Return -1 as label to indicate unknown/OOD sample
        # Data stays on CPU
        return image, -1
    
    def unload(self):
        """Explicitly unload dataset from memory"""
        self._dataset = None
        gc.collect()


class MixedTaskDataset(Dataset):
    """
    Dataset that mixes task-specific data with unknown samples
    Memory-efficient version that doesn't preload data
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
        self.unknown_size = int(self.task_size/num_classes_per_task)*2
        
        # Sample indices from unknown dataset (just indices, not data)
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


class TaskDataset(Dataset):
    """
    Wrapper dataset for a specific task.
    Memory-efficient version with lazy index building
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
        
        # Lazy index building
        self._indices = None
        self._built_indices = False
    
    def _build_indices(self):
        """Build indices only when first needed"""
        if self._built_indices:
            return
            
        self._indices = []
        
        # Try fast path first
        if hasattr(self.base_dataset, 'targets'):
            for idx, label in enumerate(self.base_dataset.targets):
                if label in self.task_classes:
                    self._indices.append(idx)
        else:
            # Slower path - iterate through dataset
            print(f"Building indices for {len(self.task_classes)} classes...")
            for idx in range(len(self.base_dataset)):
                _, label = self.base_dataset[idx]
                if label in self.task_classes:
                    self._indices.append(idx)
        
        self._built_indices = True
    
    def __len__(self) -> int:
        self._build_indices()
        return len(self._indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        self._build_indices()
        base_idx = self._indices[idx]
        image, label = self.base_dataset[base_idx]
        
        if self.remap_labels:
            label = self.label_map[label]
        
        # Data remains on CPU
        return image, label


class EnhancedContinualCIFAR100:
    """
    Enhanced CIFAR-100 dataset with SVHN unknown data for continual learning.
    Memory-efficient version with lazy loading and CPU storage.
    """
    
    def __init__(
        self,
        data_root: str = "./data",
        num_tasks: int = 10,
        classes_per_task: int = 10,
        validation_split: float = 0.1,
        unknown_ratio: float = 0.2,
        unknown_ratio_decay: float = 0.9,
        use_unknown_data: bool = True,
        seed: int = 42,
        cache_current_task_only: bool = True  # New parameter
    ):
        self.data_root = data_root
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task
        self.validation_split = validation_split
        self.unknown_ratio = unknown_ratio
        self.unknown_ratio_decay = unknown_ratio_decay
        self.use_unknown_data = use_unknown_data
        self.cache_current_task_only = cache_current_task_only
        
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
        
        # Transform for SVHN
        self.svhn_train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4377, 0.4438, 0.4728],
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
        
        # Don't load datasets upfront
        self._train_dataset = None
        self._test_dataset = None
        self._unknown_train_dataset = None
        self._unknown_test_dataset = None
        
        # Cache for current task only
        self._current_task_cache = {}
        self._current_task_id = None
        
        # Create task splits
        self.task_classes = self._create_task_splits()
    
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
    
    def _get_train_dataset(self):
        """Lazy load train dataset"""
        if self._train_dataset is None:
            print("Loading CIFAR-100 train dataset...")
            self._train_dataset = datasets.CIFAR100(
                root=self.data_root,
                train=True,
                download=True,
                transform=self.train_transform
            )
        return self._train_dataset
    
    def _get_test_dataset(self):
        """Lazy load test dataset"""
        if self._test_dataset is None:
            print("Loading CIFAR-100 test dataset...")
            self._test_dataset = datasets.CIFAR100(
                root=self.data_root,
                train=False,
                download=True,
                transform=self.test_transform
            )
        return self._test_dataset
    
    def _get_unknown_train_dataset(self):
        """Lazy load unknown train dataset"""
        if self._unknown_train_dataset is None and self.use_unknown_data:
            print("Loading SVHN train dataset as unknown...")
            self._unknown_train_dataset = SVHNUnknownDataset(
                data_root=self.data_root,
                split="train",
                transform=self.svhn_train_transform
            )
        return self._unknown_train_dataset
    
    def _get_unknown_test_dataset(self):
        """Lazy load unknown test dataset"""
        if self._unknown_test_dataset is None and self.use_unknown_data:
            print("Loading SVHN test dataset as unknown...")
            self._unknown_test_dataset = SVHNUnknownDataset(
                data_root=self.data_root,
                split="test",
                transform=self.svhn_test_transform
            )
        return self._unknown_test_dataset
    
    def clear_cache(self):
        """Clear all cached data to free memory"""
        self._current_task_cache.clear()
        self._current_task_id = None
        
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def unload_all_datasets(self):
        """Completely unload all datasets from memory"""
        self._train_dataset = None
        self._test_dataset = None
        
        if self._unknown_train_dataset:
            self._unknown_train_dataset.unload()
            self._unknown_train_dataset = None
        
        if self._unknown_test_dataset:
            self._unknown_test_dataset.unload()
            self._unknown_test_dataset = None
        
        self.clear_cache()
        print("All datasets unloaded from memory")
    
    def get_task_unknown_ratio(self, task_id: int) -> float:
        """Calculate unknown ratio for a specific task"""
        if not self.use_unknown_data:
            return 0.0
        
        return self.unknown_ratio
    
    def get_task_dataset(
        self,
        task_id: int,
        split: str = "train",
        include_unknown: bool = True
    ) -> Dataset:
        """Get dataset for a specific task with optional unknown samples"""
        
        if task_id >= self.num_tasks:
            raise ValueError(f"Task {task_id} out of range")
        
        # Use cache if available and enabled
        cache_key = f"{task_id}_{split}_{include_unknown}"
        if self.cache_current_task_only and self._current_task_id == task_id:
            if cache_key in self._current_task_cache:
                return self._current_task_cache[cache_key]
        elif self.cache_current_task_only and self._current_task_id != task_id:
            # Clear previous task cache
            self.clear_cache()
            self._current_task_id = task_id
        
        classes = self.task_classes[task_id]
        
        # Get base task dataset (lazy loaded)
        if split == "train":
            task_dataset = TaskDataset(
                self._get_train_dataset(),
                classes,
                remap_labels=True
            )
            unknown_dataset = self._get_unknown_train_dataset() if self.use_unknown_data else None
        else:
            task_dataset = TaskDataset(
                self._get_test_dataset(),
                classes,
                remap_labels=True
            )
            unknown_dataset = self._get_unknown_test_dataset() if self.use_unknown_data else None
        
        # Mix with unknown data if requested
        if include_unknown and self.use_unknown_data and unknown_dataset is not None:
            unknown_ratio = self.get_task_unknown_ratio(task_id)
            dataset = MixedTaskDataset(
                task_dataset=task_dataset,
                unknown_dataset=unknown_dataset,
                unknown_ratio=unknown_ratio,
                num_classes_per_task=self.classes_per_task
            )
        else:
            dataset = task_dataset
        
        # Cache if enabled
        if self.cache_current_task_only:
            self._current_task_cache[cache_key] = dataset
        
        return dataset
    
    def get_task_loaders(
        self,
        task_id: int,
        batch_size: int = 128,
        num_workers: int = 4,
        include_unknown_train: bool = True,
        include_unknown_test: bool = False,
        pin_memory: bool = False  # Default to False for memory efficiency
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get train, val, test loaders for a task.
        Memory-efficient version with CPU data loading.
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
                # Keep on CPU - training loop will move to GPU
                return (
                    torch.stack(images),
                    torch.tensor(labels, dtype=torch.long),
                    torch.tensor(unknown_flags, dtype=torch.float)
                )
            else:
                # Standard dataset
                images, labels = zip(*batch)
                return torch.stack(images), torch.tensor(labels, dtype=torch.long)
        
        # Create loaders with memory-efficient settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=False,  # Don't keep workers alive between epochs
            collate_fn=collate_fn if include_unknown_train and self.use_unknown_data else None
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=False,
            collate_fn=collate_fn if include_unknown_train and self.use_unknown_data else None
        )
        
        test_loader = DataLoader(
            task_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=False,
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
            'unknown_enabled': self.use_unknown_data
        }
        
        if self.use_unknown_data:
            task_train = self.get_task_dataset(task_id, "train", include_unknown=False)
            stats['task_samples'] = len(task_train)
            stats['expected_unknown_samples'] = int(len(task_train)/self.task_classes[task_id])*2
        
        return stats
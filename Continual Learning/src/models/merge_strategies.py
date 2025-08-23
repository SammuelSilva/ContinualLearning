"""
Advanced merging strategies for hierarchical LoRA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from src.models.orthogonal_utils import OrthogonalLoRAMerger, compute_orthogonality_score


class AdaptiveMergeStrategy:
    """
    Adaptive merging that decides when and how to merge based on task similarity.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.7,
        min_tasks_for_merge: int = 3,
        max_tasks_for_merge: int = 7
    ):
        self.similarity_threshold = similarity_threshold
        self.min_tasks = min_tasks_for_merge
        self.max_tasks = max_tasks_for_merge
        self.merger = OrthogonalLoRAMerger()
    
    def should_merge(
        self,
        task_adapters: Dict,
        task_performances: Dict[str, float]
    ) -> bool:
        """
        Decide if current tasks should be merged.
        
        Args:
            task_adapters: Current task adapters
            task_performances: Performance metrics for each task
            
        Returns:
            Boolean indicating whether to merge
        """
        num_tasks = len(task_adapters)
        
        # Check task count constraints
        if num_tasks < self.min_tasks:
            return False
        if num_tasks >= self.max_tasks:
            return True
        
        # Check performance stability
        perf_values = list(task_performances.values())
        perf_variance = np.var(perf_values)
        
        # Low variance suggests similar tasks - good for merging
        if perf_variance < 0.01:  # Less than 1% variance
            return True
        
        # Check orthogonality score
        adapters = list(task_adapters.values())
        ortho_score = compute_orthogonality_score(adapters)
        
        # High orthogonality means tasks are different - wait for more
        if ortho_score > 0.9:
            return False
        
        return True
    
    def select_merge_strategy(
        self,
        task_adapters: Dict,
        memory_constraint: Optional[int] = None
    ) -> str:
        """
        Select optimal merge strategy based on task characteristics.
        
        Returns:
            Strategy name: 'qr', 'svd', 'gram_schmidt', or 'blockwise'
        """
        num_tasks = len(task_adapters)
        adapters = list(task_adapters.values())
        
        # Compute characteristics
        ortho_score = compute_orthogonality_score(adapters)
        total_rank = sum(a.rank for a in adapters)
        
        # Decision logic
        if memory_constraint and total_rank > memory_constraint:
            # Need compression - use SVD
            return 'svd'
        elif ortho_score > 0.8:
            # Already quite orthogonal - simple QR is enough
            return 'qr'
        elif num_tasks <= 3:
            # Few tasks - can afford block diagonal
            return 'blockwise'
        else:
            # General case - Gram-Schmidt
            return 'gram_schmidt'


class ProgressiveMergeScheduler:
    """
    Schedules merging operations throughout training.
    """
    
    def __init__(
        self,
        initial_block_size: int = 3,
        growth_factor: float = 1.5,
        max_block_size: int = 10
    ):
        self.initial_block_size = initial_block_size
        self.growth_factor = growth_factor
        self.max_block_size = max_block_size
        self.current_block_size = initial_block_size
        self.merge_history = []
    
    def get_next_merge_point(self, current_task: int) -> int:
        """
        Determine when the next merge should occur.
        
        Args:
            current_task: Current task index
            
        Returns:
            Task index at which to perform next merge
        """
        if not self.merge_history:
            return self.initial_block_size
        
        last_merge = self.merge_history[-1]
        tasks_since_merge = current_task - last_merge
        
        # Adaptive block size
        if tasks_since_merge >= self.current_block_size:
            # Increase block size for next merge
            self.current_block_size = min(
                int(self.current_block_size * self.growth_factor),
                self.max_block_size
            )
            return current_task
        
        return last_merge + self.current_block_size
    
    def record_merge(self, task_index: int, merge_info: Dict):
        """Record a merge operation."""
        self.merge_history.append(task_index)


class TaskSimilarityAnalyzer:
    """
    Analyzes similarity between tasks to guide merging decisions.
    """
    
    @staticmethod
    def compute_gradient_similarity(
        task1_grads: Dict[str, torch.Tensor],
        task2_grads: Dict[str, torch.Tensor]
    ) -> float:
        """
        Compute cosine similarity between task gradients.
        """
        similarity_scores = []
        
        for key in task1_grads.keys():
            if key in task2_grads:
                grad1 = task1_grads[key].flatten()
                grad2 = task2_grads[key].flatten()
                
                cos_sim = F.cosine_similarity(
                    grad1.unsqueeze(0),
                    grad2.unsqueeze(0)
                ).item()
                
                similarity_scores.append(cos_sim)
        
        return np.mean(similarity_scores) if similarity_scores else 0.0
    
    @staticmethod
    def compute_feature_similarity(
        task1_features: torch.Tensor,
        task2_features: torch.Tensor
    ) -> float:
        """
        Compute similarity between task feature representations.
        """
        # Compute centroids
        centroid1 = task1_features.mean(dim=0)
        centroid2 = task2_features.mean(dim=0)
        
        # Cosine similarity between centroids
        similarity = F.cosine_similarity(
            centroid1.unsqueeze(0),
            centroid2.unsqueeze(0)
        ).item()
        
        return similarity
    
    @staticmethod
    def cluster_tasks(
        task_features: Dict[str, torch.Tensor],
        num_clusters: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Cluster tasks based on their feature representations.
        
        Returns:
            Dictionary mapping task_id to cluster_id
        """
        from sklearn.cluster import KMeans, AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        
        # Extract centroids
        centroids = []
        task_ids = []
        for task_id, features in task_features.items():
            centroid = features.mean(dim=0).cpu().numpy()
            centroids.append(centroid)
            task_ids.append(task_id)
        
        centroids = np.array(centroids)
        
        # Determine optimal number of clusters if not specified
        if num_clusters is None:
            scores = []
            for k in range(2, min(len(task_ids), 6)):
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(centroids)
                score = silhouette_score(centroids, labels)
                scores.append((k, score))
            
            num_clusters = max(scores, key=lambda x: x[1])[0]
        
        # Perform clustering
        clustering = AgglomerativeClustering(n_clusters=num_clusters)
        cluster_labels = clustering.fit_predict(centroids)
        
        # Create mapping
        task_to_cluster = {
            task_id: int(cluster_id)
            for task_id, cluster_id in zip(task_ids, cluster_labels)
        }
        
        return task_to_cluster
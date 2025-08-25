"""
Utilities for orthogonal composition of LoRA adapters.
Implements multiple strategies for merging adapters while maintaining orthogonality.
"""

from sqlite3 import adapters
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np
from src.models.lora_layers import LoRALayer

class OrthogonalProjector:
    """
    Manages orthogonal projection of LoRA adapters.
    """

    @staticmethod
    def extract_lora_layers(adapters: List) -> List[LoRALayer]:
        """
        Extract LoRALayer objects from adapters (which might be TaskSpecificLoRA).
        """
        lora_layers = []
        
        for adapter in adapters:
            if hasattr(adapter, 'lora_B') and hasattr(adapter, 'lora_A'):
                # It's already a LoRALayer
                lora_layers.append(adapter)
            elif hasattr(adapter, 'attention_adapters'):
                # It's a TaskSpecificLoRA - extract first attention LoRA for simplicity
                for attn_adapter in adapter.attention_adapters:
                    if attn_adapter is not None and hasattr(attn_adapter, 'lora_modules'):
                        # Get first available LoRA module
                        for lora_module in attn_adapter.lora_modules.values():
                            if hasattr(lora_module, 'lora_B') and hasattr(lora_module, 'lora_A'):
                                lora_layers.append(lora_module)
                                break
                    break  # Only take first layer for orthogonalization

        return lora_layers

    @staticmethod
    def qr_orthogonalization(adapters: List) -> List[LoRALayer]:
        """
        Apply QR decomposition for orthogonal composition of LoRA adapters.
        Simplified version that handles dimension mismatches better.
        """
        # Extract actual LoRA layers
        lora_adapters = OrthogonalProjector.extract_lora_layers(adapters)
        
        if not lora_adapters:
            return []
        
        # If only one adapter, return as is
        if len(lora_adapters) == 1:
            return lora_adapters
        
        orthogonal_adapters = []
        
        # Get dimensions
        first_adapter = lora_adapters[0]
        rank = first_adapter.rank
        in_features = first_adapter.lora_A.shape[1]
        out_features = first_adapter.lora_B.shape[0]
        
        # Create orthogonal vectors using Gram-Schmidt process
        A_matrices = []
        for adapter in lora_adapters:
            A_matrices.append(adapter.lora_A.data.clone())
        
        # Orthogonalize A matrices using Gram-Schmidt
        orthogonal_As = []
        for i, A in enumerate(A_matrices):
            if i == 0:
                # First matrix - just normalize
                A_orth = A / (torch.norm(A) + 1e-8)
            else:
                # Orthogonalize against all previous matrices
                A_orth = A.clone()
                for j in range(i):
                    prev_A = orthogonal_As[j]
                    # Remove projection onto previous matrix
                    for row in range(rank):
                        projection = torch.dot(A_orth[row], prev_A[row]) / (torch.dot(prev_A[row], prev_A[row]) + 1e-8)
                        A_orth[row] = A_orth[row] - projection * prev_A[row]
                
                # Normalize
                A_orth = A_orth / (torch.norm(A_orth, dim=1, keepdim=True) + 1e-8)
            
            orthogonal_As.append(A_orth)
        
        # Create new adapters with orthogonal A matrices
        for i, (A_orth, original_adapter) in enumerate(zip(orthogonal_As, lora_adapters)):
            new_adapter = LoRALayer(
                in_features=in_features,
                out_features=out_features,
                rank=rank,
                alpha=original_adapter.alpha,
                dropout=0.1
            )
            
            new_adapter.lora_A.data = A_orth
            new_adapter.lora_B.data = original_adapter.lora_B.data.clone()
            
            orthogonal_adapters.append(new_adapter)
        
        return orthogonal_adapters
    
    @staticmethod
    def svd_merge(adapters: List, target_rank: int) -> LoRALayer:
        """
        Merge multiple LoRA adapters using SVD compression.
        
        Args:
            adapters: List of LoRALayer or TaskSpecificLoRA adapters
            target_rank: Target rank for merged adapter
            
        Returns:
            Single merged LoRA adapter
        """
        # Extract actual LoRA layers
        lora_layers = OrthogonalProjector.extract_lora_layers(adapters)
        
        if not lora_layers:
            return None
        
        # Compute combined weight update
        combined_weight = torch.zeros(
            lora_layers[0].lora_B.shape[0],
            lora_layers[0].lora_A.shape[1]
        ).to(lora_layers[0].lora_A.device)
        
        for adapter in lora_layers:
            weight_update = adapter.lora_B @ adapter.lora_A * adapter.scaling
            combined_weight += weight_update / len(lora_layers)
        
        # Apply SVD
        U, S, Vt = torch.linalg.svd(combined_weight, full_matrices=False)
        
        # Truncate to target rank
        U_truncated = U[:, :target_rank]
        S_truncated = S[:target_rank]
        Vt_truncated = Vt[:target_rank, :]
        
        # Create merged adapter
        merged_adapter = LoRALayer(
            in_features=lora_layers[0].lora_A.shape[1],
            out_features=lora_layers[0].lora_B.shape[0],
            rank=target_rank,
            alpha=target_rank,
            dropout=0.1
        )
        
        # Set weights from SVD
        merged_adapter.lora_B.data = U_truncated @ torch.diag(torch.sqrt(S_truncated))
        merged_adapter.lora_A.data = torch.diag(torch.sqrt(S_truncated)) @ Vt_truncated
        
        return merged_adapter

    @staticmethod
    def gram_schmidt_orthogonalization(matrices: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply Gram-Schmidt process to orthogonalize a list of matrices.
        
        Args:
            matrices: List of weight matrices to orthogonalize
            
        Returns:
            List of orthogonalized matrices
        """
        orthogonal_matrices = []
        
        for i, matrix in enumerate(matrices):
            if i == 0:
                # First matrix: just normalize
                orthogonal_matrices.append(matrix)
            else:
                # Project out components from previous matrices
                proj = matrix.clone()
                for j in range(i):
                    # Remove projection onto previous orthogonal matrices
                    prev = orthogonal_matrices[j]
                    proj = proj - (torch.sum(proj * prev) / torch.sum(prev * prev)) * prev
                orthogonal_matrices.append(proj)
        
        return orthogonal_matrices

class OrthogonalLoRAMerger:
    """
    Advanced merging strategies for LoRA adapters.
    """
    
    def __init__(self, merge_strategy: str = "qr"):
        """
        Args:
            merge_strategy: One of 'qr', 'svd', 'gram_schmidt', 'blockwise'
        """
        self.merge_strategy = merge_strategy
        self.projector = OrthogonalProjector()
        if torch.cuda.is_available():
            self.to(torch.device('cuda:0'))
    
    def merge_adapters(
        self,
        adapters: Dict[str, nn.Module],
        strategy_params: Optional[Dict] = None
    ) -> nn.Module:
        """
        Merge multiple adapters using specified strategy.
        
        Args:
            adapters: Dictionary of task_id -> LoRA adapter (can be LoRALayer or TaskSpecificLoRA)
            strategy_params: Additional parameters for merging strategy
            
        Returns:
            Merged orthogonal adapter module
        """
        adapter_list = list(adapters.values())
        
        # Check adapter type and handle accordingly
        if adapter_list and hasattr(adapter_list[0], 'attention_adapters'):
            # These are TaskSpecificLoRA objects - we need to handle differently
            return self._merge_task_specific_loras(adapters, strategy_params)
        
        # Original implementation for LoRALayer objects
        if self.merge_strategy == "qr":
            return self._qr_merge(adapter_list)
        elif self.merge_strategy == "svd":
            target_rank = strategy_params.get('target_rank', 16) if strategy_params else 16
            return self._svd_merge(adapter_list, target_rank)
        elif self.merge_strategy == "gram_schmidt":
            return self._gram_schmidt_merge(adapter_list)
        elif self.merge_strategy == "blockwise":
            return self._blockwise_merge(adapter_list)
        else:
            raise ValueError(f"Unknown merge strategy: {self.merge_strategy}")

    def _merge_task_specific_loras(self, adapters: Dict[str, nn.Module], strategy_params: Optional[Dict] = None):
            """
            Special handling for merging TaskSpecificLoRA objects.
            """
            # For TaskSpecificLoRA, we don't actually merge them here
            # The OrthogonalMergedBlock will handle the layer-wise merging
            # Just return a placeholder that indicates these should be handled specially
            
            class TaskSpecificLoRAContainer(nn.Module):
                def __init__(self, task_adapters):
                    super().__init__()
                    self.task_adapters = nn.ModuleDict(task_adapters)
                    self.is_task_specific = True
                
                def forward(self, x):
                    # This is a placeholder - actual forward is handled by OrthogonalMergedBlock
                    return x
            
            return TaskSpecificLoRAContainer(adapters)
    
    def _qr_merge(self, adapters: List[LoRALayer]) -> nn.Module:
        """QR-based orthogonal merge."""
        orthogonal_adapters = self.projector.qr_orthogonalization(adapters)
        return CombinedOrthogonalAdapter(orthogonal_adapters)
    
    def _svd_merge(self, adapters: List[LoRALayer], target_rank: int) -> nn.Module:
        """SVD-based compression merge."""
        return self.projector.svd_merge(adapters, target_rank)
    
    def _gram_schmidt_merge(self, adapters: List[LoRALayer]) -> nn.Module:
        """Gram-Schmidt orthogonalization merge."""
        # Extract weight matrices
        weights = [adapter.lora_B @ adapter.lora_A for adapter in adapters]
        orthogonal_weights = self.projector.gram_schmidt_orthogonalization(weights)
        
        # Create combined adapter
        return CombinedOrthogonalAdapter(adapters, orthogonal_weights)
    
    def _blockwise_merge(self, adapters: List[LoRALayer]) -> nn.Module:
        """
        Block diagonal merge - maintains complete independence.
        Most memory-intensive but guarantees zero interference.
        """
        return BlockDiagonalAdapter(adapters)


class CombinedOrthogonalAdapter(nn.Module):
    """
    Combined adapter that applies multiple orthogonal LoRA adapters.
    """
    
    def __init__(self, adapters: List[LoRALayer], weights: Optional[List[torch.Tensor]] = None):
        super().__init__()
        self.adapters = nn.ModuleList(adapters)
        self.num_adapters = len(adapters)
        
        if weights is not None:
            self.register_buffer('orthogonal_weights', torch.stack(weights))
        else:
            self.orthogonal_weights = None
        
        # Task routing weights (learnable)
        self.task_gates = nn.Parameter(torch.ones(self.num_adapters) / self.num_adapters)
        if torch.cuda.is_available():
            self.to(torch.device('cuda:0'))

    def forward(self, x: torch.Tensor, task_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through combined adapter.
        
        Args:
            x: Input tensor
            task_indices: Optional task indices for routing
        """
        if task_indices is not None:
            # Task-specific routing
            output = torch.zeros_like(x)
            for i, adapter in enumerate(self.adapters):
                mask = (task_indices == i).float().unsqueeze(-1)
                output += adapter(x) * mask
        else:
            # Weighted combination
            output = torch.zeros_like(x)
            gates = F.softmax(self.task_gates, dim=0)
            for i, adapter in enumerate(self.adapters):
                output += adapter(x) * gates[i]
        
        return output

    def create_lora_from_svd(U: torch.Tensor, S: torch.Tensor, Vt: torch.Tensor, rank: int) -> LoRALayer:
        """
        Create a LoRA layer from SVD decomposition.
        """
        in_features = Vt.shape[1]
        out_features = U.shape[0]
        
        lora = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=rank,
            dropout=0.1
        )
        
        # Set weights from SVD
        lora.lora_A.data = Vt[:rank, :]
        lora.lora_B.data = U[:, :rank] @ torch.diag(S[:rank])
        
        return lora

    def project_to_null_space(matrix: torch.Tensor, null_space_basis: torch.Tensor) -> torch.Tensor:
        """
        Project a matrix to the null space of another.
        """
        # Project out the components in the given space
        projection = null_space_basis @ null_space_basis.T @ matrix
        return matrix - projection

    def compute_null_space(matrices: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute the null space of a set of matrices.
        """
        # Stack matrices
        combined = torch.cat([m.flatten().unsqueeze(0) for m in matrices], dim=0)
        
        # SVD to find null space
        _, _, Vt = torch.linalg.svd(combined, full_matrices=True)
        
        # Null space is the right singular vectors corresponding to zero singular values
        rank = torch.linalg.matrix_rank(combined)
        null_space = Vt[rank:].T
        
        return null_space

    def compute_projection(new_lora: nn.Module, prev_lora: nn.Module) -> nn.Module:
        """
        Compute projection of one LoRA onto another.
        """
        # Get weight matrices
        W_new = new_lora.lora_B @ new_lora.lora_A
        W_prev = prev_lora.lora_B @ prev_lora.lora_A
        
        # Compute projection
        W_new_flat = W_new.flatten()
        W_prev_flat = W_prev.flatten()
        
        projection_scalar = torch.dot(W_new_flat, W_prev_flat) / torch.dot(W_prev_flat, W_prev_flat)
        
        # Create projected LoRA
        projected_lora = LoRALayer(
            in_features=new_lora.lora_A.shape[1],
            out_features=new_lora.lora_B.shape[0],
            rank=new_lora.rank,
            alpha=new_lora.alpha,
            dropout=0.1
        )
        
        # Set projected weights
        projected_weight = projection_scalar * W_prev
        U, S, Vt = torch.linalg.svd(projected_weight, full_matrices=False)
        
        rank = new_lora.rank
        projected_lora.lora_A.data = Vt[:rank, :]
        projected_lora.lora_B.data = U[:, :rank] @ torch.diag(S[:rank].sqrt())
        
        return projected_lora

class BlockDiagonalAdapter(nn.Module):
    """
    Block diagonal adapter that maintains complete task separation.
    """
    
    def __init__(self, adapters: List[LoRALayer]):
        super().__init__()
        self.adapters = nn.ModuleList(adapters)
        self.num_tasks = len(adapters)
        
        # Compute block boundaries
        self.ranks = [adapter.rank for adapter in adapters]
        self.cumsum_ranks = torch.tensor([0] + list(np.cumsum(self.ranks)))
        self.total_rank = sum(self.ranks)
        
        # Create block diagonal matrices
        self._create_block_diagonal()

        if torch.cuda.is_available():
            self.to(torch.device('cuda:0'))

    def _create_block_diagonal(self):
        """Create block diagonal structure."""
        device = self.adapters[0].lora_A.device
        
        # Initialize block diagonal A matrix
        total_in = self.adapters[0].lora_A.shape[1]
        self.block_A = torch.zeros(self.total_rank, total_in).to(device)
        
        # Initialize block diagonal B matrix
        total_out = self.adapters[0].lora_B.shape[0]
        self.block_B = torch.zeros(total_out, self.total_rank).to(device)
        
        # Fill blocks
        for i, adapter in enumerate(self.adapters):
            start_idx = self.cumsum_ranks[i]
            end_idx = self.cumsum_ranks[i + 1]
            
            self.block_A[start_idx:end_idx] = adapter.lora_A.data
            self.block_B[:, start_idx:end_idx] = adapter.lora_B.data
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through block diagonal structure."""
        return (x @ self.block_A.T) @ self.block_B.T
    
    def get_task_output(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """Get output for specific task."""
        return self.adapters[task_id](x)


def compute_orthogonality_score(adapters: List) -> float:
    """
    Compute orthogonality score between adapters.
    Score of 1.0 means perfect orthogonality.
    
    Args:
        adapters: List of LoRALayer or TaskSpecificLoRA objects
    """
    if len(adapters) < 2:
        return 1.0
    
    # Extract actual LoRA layers based on adapter type
    lora_layers = []
    
    for adapter in adapters:
        if hasattr(adapter, 'lora_B') and hasattr(adapter, 'lora_A'):
            # It's a LoRALayer
            lora_layers.append(adapter)
        elif hasattr(adapter, 'attention_adapters') or hasattr(adapter, 'ffn_adapters'):
            # It's a TaskSpecificLoRA - extract individual LoRA layers
            # Get LoRA layers from attention adapters
            if hasattr(adapter, 'attention_adapters'):
                for attn_adapter in adapter.attention_adapters:
                    if attn_adapter is not None and hasattr(attn_adapter, 'lora_modules'):
                        for lora_module in attn_adapter.lora_modules.values():
                            if hasattr(lora_module, 'lora_B') and hasattr(lora_module, 'lora_A'):
                                lora_layers.append(lora_module)
            
            # Get LoRA layers from FFN adapters
            if hasattr(adapter, 'ffn_adapters'):
                for ffn_adapter in adapter.ffn_adapters:
                    if ffn_adapter is not None:
                        if hasattr(ffn_adapter, 'lora_fc1'):
                            lora_layers.append(ffn_adapter.lora_fc1)
                        if hasattr(ffn_adapter, 'lora_fc2'):
                            lora_layers.append(ffn_adapter.lora_fc2)
    
    if len(lora_layers) < 2:
        return 1.0
    
    # Compute weight updates
    weights = []
    for lora in lora_layers:
        if hasattr(lora, 'lora_B') and hasattr(lora, 'lora_A'):
            W = (lora.lora_B @ lora.lora_A).flatten()
            W_normalized = W / (torch.norm(W) + 1e-8)
            weights.append(W_normalized)
    
    if len(weights) < 2:
        return 1.0
    
    # Compute pairwise dot products
    orthogonality_scores = []
    for i in range(len(weights)):
        for j in range(i + 1, len(weights)):
            dot_product = torch.abs(torch.dot(weights[i], weights[j]))
            orthogonality_scores.append(1.0 - dot_product.item())
    
    return np.mean(orthogonality_scores) if orthogonality_scores else 1.0
"""
Hierarchical LoRA management with orthogonal merging capability.
Extends the existing ContinualLoRAViT with block-based organization.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from src.models.lora_vit import ContinualLoRAViT, LoRAAttention, LoRAMlp
from src.models.lora_layers import TaskSpecificLoRA
from src.models.orthogonal_utils import BlockDiagonalAdapter, CombinedOrthogonalAdapter

class OrthogonalMergedBlock(nn.Module):
    """
    Represents a merged block of orthogonal LoRA adapters.
    Now properly uses the utility functions from orthogonal_utils.py
    """
    
    def __init__(
        self,
        original_blocks: List[Dict[str, nn.Module]],
        task_adapters: Dict[str, TaskSpecificLoRA],
        task_heads: Dict[str, nn.Module],
        block_id: int,
        hidden_dim: int,
        num_layers: int,
        mlp_dim: int,
        num_heads: int,
        lora_rank: int,
        lora_alpha: float,
        lora_config: str = "attention_only",
        merge_strategy: str = "qr"  # Add merge strategy parameter
    ):
        super().__init__()
        self.original_blocks = original_blocks
        self.block_id = block_id
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_config = lora_config
        self.merge_strategy = merge_strategy
        
        # Import the utilities we created
        from src.models.orthogonal_utils import (
            OrthogonalLoRAMerger,
            compute_orthogonality_score,
            OrthogonalProjector
        )
        
        self.merger = OrthogonalLoRAMerger(merge_strategy=merge_strategy)
        self.projector = OrthogonalProjector()
        
        # Store original task adapters (frozen)
        self.task_adapters = nn.ModuleDict(task_adapters)

        self.task_heads = nn.ModuleDict()
        for task_id, head in task_heads.items():
            self.task_heads[task_id] = head
            print(f"DEBUG: Added head for {task_id} to merged block")
            
        self.task_ids = list(task_adapters.keys())
        self.task_to_index = {tid: i for i, tid in enumerate(self.task_ids)}
        
        # Compute and log orthogonality score
        adapter_list = list(task_adapters.values())
        self.orthogonality_score = compute_orthogonality_score(adapter_list)
        print(f"  Block {block_id} orthogonality score: {self.orthogonality_score:.3f}")
        
        # Create merged orthogonal LoRA modules using our utilities
        self._create_injectable_lora_modules()
        
        # Freeze all parameters in this block
        for param in self.parameters():
            param.requires_grad = False
        
        # Create block-level unknown head
        self.block_unknown_head = nn.Linear(hidden_dim, 2)
        
        if torch.cuda.is_available():
            self.to(torch.device('cuda:0'))

    def compute_block_confidence(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence that sample belongs to this block.
        
        Args:
            features: Pre-computed features from backbone (shape: [batch_size, hidden_dim])
            
        Returns:
            Confidence scores (probability of being in-block) (shape: [batch_size])
        """
        # Use the block-level unknown head to compute confidence
        logits = self.block_unknown_head(features)  # Shape: [batch_size, 2]
        
        # Apply softmax to get probabilities
        # Index 0: probability of being in-block
        # Index 1: probability of being unknown/out-of-block
        confidence = F.softmax(logits, dim=-1)[:, 0]  # Shape: [batch_size]
        
        return confidence
    
    def compute_task_unknown_probabilities(self, x: torch.Tensor, backbone: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Compute unknown probabilities for each task in the block.
        
        Args:
            x: Input images (shape: [batch_size, 3, H, W])
            backbone: Reference to the ViT backbone
            
        Returns:
            Dictionary mapping task_id to unknown probability tensor
        """
        unknown_probs = {}
        
        for task_id in self.task_ids:
            # Get features for this task using the merged adapters
            features = self.forward_with_backbone(x, backbone, task_id)
            
            # Apply task head to get logits
            logits = self.task_heads[task_id](features)
            
            # Get unknown probability (last class)
            probs = F.softmax(logits, dim=-1)
            unknown_probs[task_id] = probs[:, -1]  # Shape: [batch_size]
        
        return unknown_probs
    
    def forward_with_backbone(self, x: torch.Tensor, backbone: nn.Module, 
                             task_id: Optional[str] = None) -> torch.Tensor:
        """
        Forward pass through backbone with this block's LoRA adapters.
        
        Args:
            x: Input images
            backbone: ViT backbone
            task_id: Optional task ID for task-specific routing
            
        Returns:
            Features from backbone with LoRA adaptation
        """
        # Inject our adapters
        self.inject_into_backbone(backbone, task_id)
        
        try:
            # Forward through modified backbone
            features = backbone(x)
        finally:
            # Always remove our adapters
            self.remove_from_backbone(backbone)
        
        return features
    
    def get_task_head(self, task_id: str):
        """Get specific task head from block"""
        # ModuleDict doesn't have .get(), use item access
        if task_id in self.task_heads:
            return self.task_heads[task_id]
        return None

    def get_task_adapter(self, task_id: str):
        """Get specific task adapter from block"""
        # ModuleDict doesn't have .get(), use item access
        if task_id in self.task_adapters:
            return self.task_adapters[task_id]
        return None

    def _create_injectable_lora_modules(self):
        """
        Create LoRA modules using the utility functions we defined.
        """
        self.merged_attention_modules = nn.ModuleList()
        self.merged_ffn_modules = nn.ModuleList()
        
        for layer_idx in range(self.num_layers):
            # Create merged attention adapter using utilities
            if self.lora_config in ["attention_only", "both"]:
                merged_attn = self._create_merged_attention_adapter_with_utils(layer_idx)
                self.merged_attention_modules.append(merged_attn)
            else:
                self.merged_attention_modules.append(None)
            
            # Create merged FFN adapter using utilities
            if self.lora_config in ["ffn_only", "both"]:
                merged_ffn = self._create_merged_ffn_adapter_with_utils(layer_idx)
                self.merged_ffn_modules.append(merged_ffn)
            else:
                self.merged_ffn_modules.append(None)
    
    def _create_merged_attention_adapter_with_utils(self, layer_idx: int) -> nn.Module:
        """
        Create a merged attention adapter using our utility functions.
        """
        # Collect LoRA layers from all tasks for this layer
        q_loras = {}
        v_loras = {}
        k_loras = {}
        
        for task_id in self.task_ids:
            task_adapter = self.task_adapters[task_id]
            if task_adapter.attention_adapters[layer_idx] is not None:
                attn_adapter = task_adapter.attention_adapters[layer_idx]
                if 'q' in attn_adapter.lora_modules:
                    q_loras[task_id] = attn_adapter.lora_modules['q']
                if 'v' in attn_adapter.lora_modules:
                    v_loras[task_id] = attn_adapter.lora_modules['v']
                if 'k' in attn_adapter.lora_modules:
                    k_loras[task_id] = attn_adapter.lora_modules['k']
        
        # Use the merger to create orthogonal combinations
        merged_q = None
        merged_v = None
        merged_k = None
        
        if q_loras:
            q_list = list(q_loras.values())
            if self.merge_strategy == "qr" and len(q_list) > 1:
                # Use QR orthogonalization
                try:
                    orthogonal_q_list = self.projector.qr_orthogonalization(q_list)
                    merged_q = CombinedOrthogonalAdapter(orthogonal_q_list)
                except Exception as e:
                    print(f"  Warning: QR orthogonalization failed for Q: {e}")
                    # Fallback to simple combination
                    merged_q = CombinedOrthogonalAdapter(q_list)
            elif self.merge_strategy == "svd":
                # Use SVD merge
                target_rank = self.lora_rank * min(4, len(q_loras))
                merged_q = self.projector.svd_merge(q_list, target_rank)
            elif self.merge_strategy == "blockwise":
                # Use block diagonal
                merged_q = BlockDiagonalAdapter(q_list)
            else:
                # Simple combination
                merged_q = CombinedOrthogonalAdapter(q_list)
        
        if v_loras:
            v_list = list(v_loras.values())
            if self.merge_strategy == "qr" and len(v_list) > 1:
                try:
                    orthogonal_v_list = self.projector.qr_orthogonalization(v_list)
                    merged_v = CombinedOrthogonalAdapter(orthogonal_v_list)
                except Exception as e:
                    print(f"  Warning: QR orthogonalization failed for V: {e}")
                    merged_v = CombinedOrthogonalAdapter(v_list)
            elif self.merge_strategy == "svd":
                target_rank = self.lora_rank * min(4, len(v_loras))
                merged_v = self.projector.svd_merge(v_list, target_rank)
            elif self.merge_strategy == "blockwise":
                merged_v = BlockDiagonalAdapter(v_list)
            else:
                merged_v = CombinedOrthogonalAdapter(v_list)
        
        if k_loras:
            # Similar for K
            k_list = list(k_loras.values())
            if self.merge_strategy == "qr" and len(k_list) > 1:
                try:
                    orthogonal_k_list = self.projector.qr_orthogonalization(k_list)
                    merged_k = CombinedOrthogonalAdapter(orthogonal_k_list)
                except Exception as e:
                    print(f"  Warning: QR orthogonalization failed for K: {e}")
                    merged_k = CombinedOrthogonalAdapter(k_list)
            else:
                merged_k = CombinedOrthogonalAdapter(k_list) if k_list else None
        
        # Create a module that holds all merged components
        class MergedAttentionModule(nn.Module):
            def __init__(self, q_module, v_module, k_module, task_mapping):
                super().__init__()
                self.q_module = q_module
                self.v_module = v_module
                self.k_module = k_module
                self.task_mapping = task_mapping
                
                # Also store original task-specific modules for routing
                self.task_q = nn.ModuleDict(task_mapping.get('q', {}))
                self.task_v = nn.ModuleDict(task_mapping.get('v', {}))
                self.task_k = nn.ModuleDict(task_mapping.get('k', {}))
            
            def forward(self, x: torch.Tensor, task_id: Optional[str] = None) -> Dict[str, torch.Tensor]:
                outputs = {}
                
                if task_id is not None:
                    # Task-specific routing
                    if task_id in self.task_q:
                        outputs['q'] = self.task_q[task_id](x)
                    if task_id in self.task_v:
                        outputs['v'] = self.task_v[task_id](x)
                    if task_id in self.task_k:
                        outputs['k'] = self.task_k[task_id](x)
                else:
                    # Use merged modules
                    if self.q_module is not None:
                        outputs['q'] = self.q_module(x)
                    if self.v_module is not None:
                        outputs['v'] = self.v_module(x)
                    if self.k_module is not None:
                        outputs['k'] = self.k_module(x)
                
                return outputs
        
        return MergedAttentionModule(
            merged_q, merged_v, merged_k,
            {'q': q_loras, 'v': v_loras, 'k': k_loras}
        )
    
    def _create_merged_ffn_adapter_with_utils(self, layer_idx: int) -> nn.Module:
        """
        Create a merged FFN adapter using our utility functions.
        """
        fc1_loras = {}
        fc2_loras = {}
        
        for task_id in self.task_ids:
            task_adapter = self.task_adapters[task_id]
            if task_adapter.ffn_adapters[layer_idx] is not None:
                ffn_adapter = task_adapter.ffn_adapters[layer_idx]
                fc1_loras[task_id] = ffn_adapter.lora_fc1
                fc2_loras[task_id] = ffn_adapter.lora_fc2
        
        # Use merger for FFN layers
        merged_fc1 = None
        merged_fc2 = None
        
        if fc1_loras:
            strategy_params = {'target_rank': self.lora_rank * min(4, len(fc1_loras))} if self.merge_strategy == "svd" else None
            merged_fc1 = self.merger.merge_adapters(fc1_loras, strategy_params)
        
        if fc2_loras:
            strategy_params = {'target_rank': self.lora_rank * min(4, len(fc2_loras))} if self.merge_strategy == "svd" else None
            merged_fc2 = self.merger.merge_adapters(fc2_loras, strategy_params)
        
        class MergedFFNModule(nn.Module):
            def __init__(self, fc1_module, fc2_module, task_mapping):
                super().__init__()
                self.fc1_module = fc1_module
                self.fc2_module = fc2_module
                self.task_fc1 = nn.ModuleDict(task_mapping.get('fc1', {}))
                self.task_fc2 = nn.ModuleDict(task_mapping.get('fc2', {}))
            
            def forward(self, fc1_input: Optional[torch.Tensor] = None,
                       fc2_input: Optional[torch.Tensor] = None,
                       task_id: Optional[str] = None) -> Dict[str, torch.Tensor]:
                outputs = {}
                
                if task_id is not None:
                    # Task-specific
                    if fc1_input is not None and task_id in self.task_fc1:
                        outputs['fc1'] = self.task_fc1[task_id](fc1_input)
                    if fc2_input is not None and task_id in self.task_fc2:
                        outputs['fc2'] = self.task_fc2[task_id](fc2_input)
                else:
                    # Merged
                    if fc1_input is not None and self.fc1_module is not None:
                        outputs['fc1'] = self.fc1_module(fc1_input)
                    if fc2_input is not None and self.fc2_module is not None:
                        outputs['fc2'] = self.fc2_module(fc2_input)
                
                return outputs
        
        return MergedFFNModule(
            merged_fc1, merged_fc2,
            {'fc1': fc1_loras, 'fc2': fc2_loras}
        )
    
    def inject_into_backbone(self, backbone: nn.Module, task_id: Optional[str] = None):
        """
        Inject this block's merged LoRA modules into the backbone.
        Fixed to work with the existing LoRAAttention/LoRAMlp structure.
        """

        for layer_idx, block in enumerate(backbone.blocks):
            # Get the TRUE original (not LoRA-wrapped)
            original_attn = self.original_blocks[layer_idx]['attn']
            original_mlp = self.original_blocks[layer_idx]['mlp']

            print(f"  - Using original_attn type: {type(original_attn)}")
            print(f"  - original_attn.qkv type: {type(original_attn.qkv) if hasattr(original_attn, 'qkv') else 'NO QKV ATTR'}")

            # Attention
            if layer_idx < len(self.merged_attention_modules) and self.merged_attention_modules[layer_idx] is not None:
                merged_attn_module = self.merged_attention_modules[layer_idx]
                
                # Create a LoRAAttention that uses our merged adapters
                if task_id and task_id in merged_attn_module.task_q:
                    # Task-specific: use the original task's LoRA
                    lora_attn = LoRAAttention(
                        original_attn=original_attn,
                        hidden_dim=self.hidden_dim,
                        lora_rank=self.lora_rank,
                        lora_alpha=self.lora_alpha,
                        lora_dropout=0.1,
                        target_modules=["q", "v"]
                    )
                    # Use task-specific adapters
                    if task_id in merged_attn_module.task_q:
                        lora_attn.lora_adapters["q"] = merged_attn_module.task_q[task_id]
                    if task_id in merged_attn_module.task_v:
                        lora_attn.lora_adapters["v"] = merged_attn_module.task_v[task_id]
                else:
                    # Use merged adapters - but we need to wrap them properly
                    print(f"  - Using merged adapters")

                    lora_attn = LoRAAttention(
                        original_attn=original_attn,
                        hidden_dim=self.hidden_dim,
                        lora_rank=self.lora_rank,
                        lora_alpha=self.lora_alpha,
                        lora_dropout=0.1,
                        target_modules=["q", "v"]
                    )
                    print(f"  - Created LoRAAttention, checking its original_attn.qkv: {type(lora_attn.original_attn.qkv)}")

                    # Replace with merged modules
                    if merged_attn_module.q_module:
                        lora_attn.lora_adapters["q"] = merged_attn_module.q_module
                    if merged_attn_module.v_module:
                        lora_attn.lora_adapters["v"] = merged_attn_module.v_module
                
                block.attn = lora_attn
            
            # FFN (similar pattern)
            if layer_idx < len(self.merged_ffn_modules) and self.merged_ffn_modules[layer_idx] is not None:
                merged_ffn_module = self.merged_ffn_modules[layer_idx]
                
                lora_mlp = LoRAMlp(
                    original_mlp=original_mlp,
                    hidden_dim=self.hidden_dim,
                    mlp_dim=self.mlp_dim,
                    lora_rank=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=0.1
                )
                
                if task_id and task_id in merged_ffn_module.task_fc1:
                    # Task-specific
                    lora_mlp.lora_fc1 = merged_ffn_module.task_fc1[task_id]
                    lora_mlp.lora_fc2 = merged_ffn_module.task_fc2[task_id]
                else:
                    # Merged
                    if merged_ffn_module.fc1_module:
                        lora_mlp.lora_fc1 = merged_ffn_module.fc1_module
                    if merged_ffn_module.fc2_module:
                        lora_mlp.lora_fc2 = merged_ffn_module.fc2_module
                
                block.mlp = lora_mlp

    def remove_from_backbone(self, backbone: nn.Module):
        """
        Remove this block's LoRA modules from the backbone.
        """
        for layer_idx, block in enumerate(backbone.blocks):
            block.attn = self.original_blocks[layer_idx]['attn']
            block.mlp = self.original_blocks[layer_idx]['mlp']
    
    def forward_with_backbone(self, x: torch.Tensor, backbone: nn.Module, 
                             task_id: Optional[str] = None) -> torch.Tensor:
        """
        Forward pass through backbone with this block's LoRA adapters.
        """
        # Inject our adapters
        self.inject_into_backbone(backbone, task_id)
        
        try:
            # Forward through modified backbone
            features = backbone(x)
        finally:
            # Always remove our adapters
            self.remove_from_backbone(backbone)
        
        return features
    
    def forward_task(self, x: torch.Tensor, task_index: int, backbone: nn.Module) -> torch.Tensor:
        """
        Forward pass for a specific task within the block.
        """
        if task_index >= len(self.task_ids):
            raise ValueError(f"Task index {task_index} out of range")
        
        task_id = self.task_ids[task_index]
        
        print(f"DEBUG: forward_task for {task_id} (index {task_index})")
        # Get features with task-specific LoRA
        features = self.forward_with_backbone(x, backbone, task_id)
        
        # Apply task-specific head
        task_head = self.get_task_head(task_id)
        if task_head is not None:
            logits = self.task_heads[task_id](features)
            print(f"DEBUG: Logits shape for task {task_id}: {logits.shape}, max: {logits.max().item()}")
        else:
            print(f"ERROR: No task head found for task {task_id}")
            logits = torch.zeros(x.shape[0], 11, device=x.device)

        return logits
    
    def forward_all(self, x: torch.Tensor, backbone: nn.Module) -> torch.Tensor:
        """
        Forward pass through all tasks (averaged).
        """
        # Use merged adapters (no specific task_id)
        features = self.forward_with_backbone(x, backbone, task_id=None)
        return features
    
    def compute_task_unknown_probabilities(self, x: torch.Tensor, backbone: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Compute unknown probabilities for each task in the block.
        """
        unknown_probs = {}
        
        for task_id in self.task_ids:
            # Get features for this task
            features = self.forward_with_backbone(x, backbone, task_id)
            
            # Apply task head
            logits = self.task_heads[task_id](features)
            
            # Get unknown probability
            probs = F.softmax(logits, dim=-1)
            unknown_probs[task_id] = probs[:, -1]
        
        return unknown_probs

class HierarchicalLoRAViT(ContinualLoRAViT):
    """
    Hierarchical extension of ContinualLoRAViT with orthogonal block merging.
    """
    
    def __init__(
        self,
        vit_model_name: str = "vit_base_patch16_224",
        lora_rank: int = 4,
        lora_alpha: float = 4.0,
        lora_dropout: float = 0.1,
        lora_config: str = "attention_only",
        use_pretrained: bool = True,
        tasks_per_block: int = 5,  # Hard limit for merging
        use_orthogonal_merge: bool = True
    ):
        super().__init__(
            vit_model_name=vit_model_name,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_config=lora_config,
            use_pretrained=use_pretrained
        )
        
        self.tasks_per_block = tasks_per_block
        self.use_orthogonal_merge = use_orthogonal_merge
        
        # Hierarchical storage
        self.merged_blocks = nn.ModuleList()
        self.active_block_tasks = {}  # Tasks in current (unmerged) block
        self.current_block_id = 0
        
        # Block-level heads
        self.block_unknown_heads = nn.ModuleList()

        if torch.cuda.is_available():
            self.to(torch.device('cuda:0'))

    def add_task(self, task_id: str, num_classes: int):
        """Override to implement hierarchical organization"""
        
        # Check if we need to merge current block
        if len(self.active_block_tasks) >= self.tasks_per_block:
            print(f"\n📦 Merging block {self.current_block_id} with {len(self.active_block_tasks)} tasks...")
            self._merge_current_block()
            self.current_block_id += 1
            self.active_block_tasks = {}
        
        # Add task using parent method
        super().add_task(task_id, num_classes)
        
        # Track in active block
        self.active_block_tasks[task_id] = {
            'num_classes': num_classes,
            'task_index': len(self.active_block_tasks)
        }
        
        # Add block unknown head if first task in block
        if len(self.active_block_tasks) == 1:
            self.block_unknown_heads.append(
                nn.Linear(self.hidden_dim, 2).to(next(self.parameters()).device)
            )
            print(f"📊 Started new block {self.current_block_id}")
    
    def _merge_current_block(self):
        """Merge current active tasks into an orthogonal block using adaptive strategy"""
        
        # Import the strategy utilities
        from src.models.merge_strategies import (
            AdaptiveMergeStrategy,
            TaskSimilarityAnalyzer,
            ProgressiveMergeScheduler
        )
        
        # Use adaptive strategy to decide merge approach
        adaptive_strategy = AdaptiveMergeStrategy()
        
        # Collect performance metrics for adaptive decision
        task_performances = {}
        for task_id in self.active_block_tasks.keys():
            # Get last known performance (you'd track this during training)
            task_performances[task_id] = getattr(self, f'{task_id}_last_acc', 0.85)
        
        # Check if we should merge
        should_merge = adaptive_strategy.should_merge(
            self.task_adapters,
            task_performances
        )
        
        if not should_merge and len(self.active_block_tasks) < self.tasks_per_block:
            print(f"  Adaptive strategy suggests waiting for more tasks before merging")
            return
        
        # Select merge strategy
        merge_strategy = adaptive_strategy.select_merge_strategy(
            self.task_adapters,
            memory_constraint=None  # Could add memory constraints
        )
        
        print(f"  Using {merge_strategy} merge strategy")
        
        # Analyze task similarity if needed
        if hasattr(self, 'task_features'):
            analyzer = TaskSimilarityAnalyzer()
            clusters = analyzer.cluster_tasks(self.task_features)
            print(f"  Task clusters: {clusters}")
        
        # Collect adapters and heads for active tasks
        adapters_to_merge = {}
        heads_to_merge = {}
        
        for task_id in self.active_block_tasks.keys():
            adapters_to_merge[task_id] = self.task_adapters[task_id]
            heads_to_merge[task_id] = self.task_heads[task_id]
        
        # Create merged block with selected strategy
        merged_block = OrthogonalMergedBlock(
            original_blocks=self.original_blocks,
            task_adapters=adapters_to_merge,
            task_heads=heads_to_merge,
            block_id=self.current_block_id,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_config=self.lora_config,
            merge_strategy=merge_strategy  # Pass the selected strategy
        )
        
        # Log orthogonality score
        from src.models.orthogonal_utils import compute_orthogonality_score
        ortho_score = compute_orthogonality_score(list(adapters_to_merge.values()))
        print(f"  ✅ Orthogonality score: {ortho_score:.3f}")
        
        if self.use_orthogonal_merge:
            print(f"  ✅ Applied {merge_strategy} orthogonal composition")
        
        # Add to merged blocks
        self.merged_blocks.append(merged_block)
        
        # Remove from active storage
        for task_id in self.active_block_tasks.keys():
            del self.task_adapters[task_id]
            del self.task_heads[task_id]
        
        print(f"  ✅ Block {self.current_block_id} merged successfully")
            
    def predict_task_id_hierarchical(self, x: torch.Tensor) -> Tuple[List[str], torch.Tensor]:
        """
        Hierarchical task prediction with dual unknown mechanism.
        """
        batch_size = x.shape[0]
        
        with torch.no_grad():
            # Get backbone features once
            features = self.backbone(x)
            
            # Step 1: Block-level prediction
            block_confidences = []
            
            # Check merged blocks
            for block_idx, block in enumerate(self.merged_blocks):
                confidence = block.compute_block_confidence(features)
                block_confidences.append(confidence)
            
            # Check active block if it has tasks
            if self.active_block_tasks:
                active_confidence = F.softmax(
                    self.block_unknown_heads[-1](features), dim=-1
                )[:, 0]
                block_confidences.append(active_confidence)
            
            # Select most confident block for each sample
            if block_confidences:
                block_confidence_matrix = torch.stack(block_confidences, dim=1)
                max_confidences, block_indices = torch.max(block_confidence_matrix, dim=1)
            else:
                # No blocks yet, use active tasks
                block_indices = torch.zeros(batch_size, dtype=torch.long)
            
            # Step 2: Task-level prediction within selected blocks
            predicted_tasks = []
            task_confidences = []
            flag = True
            for i in range(batch_size):
                if block_confidences:
                    block_idx = block_indices[i].item()
                else:
                    block_idx = 0
                if block_idx < len(self.merged_blocks):
                    # Task is in a merged block
                    block = self.merged_blocks[block_idx]
                    
                    # Get unknown probabilities for each task in block
                    task_unknown_probs = block.compute_task_unknown_probabilities(
                        x[i:i+1], self.backbone
                    )
                    
                    # Select task with lowest unknown probability
                    best_task = min(task_unknown_probs, key=lambda k: task_unknown_probs[k].item())
                    predicted_tasks.append(best_task)
                    task_confidences.append(1 - task_unknown_probs[best_task].item())
                    
                else:
                    # Task is in active block
                    if self.active_block_tasks:
                        task_unknown_probs = {}
                        for task_id in self.active_block_tasks.keys():
                            logits = self.forward(x[i:i+1], task_id=task_id)
                            probs = F.softmax(logits, dim=-1)
                            task_unknown_probs[task_id] = probs[0, -1].item()
                        
                        best_task = min(task_unknown_probs, key=task_unknown_probs.get)
                        predicted_tasks.append(best_task)
                        task_confidences.append(1 - task_unknown_probs[best_task])
                    else:
                        # Fallback
                        print("DEBUG: Fallback.")
                        predicted_tasks.append("task_0")
                        task_confidences.append(0.0)

            task_confidences = torch.tensor(task_confidences, device=x.device)
    
        return predicted_tasks, task_confidences

    def set_active_task(self, task_id: str):
        """Override to handle hierarchical structure"""
        # First, check if task exists
        task_found = False
        
        # Check if task is in active block
        if task_id in self.task_adapters:
            # Task is in active block - use parent's method
            super().set_active_task(task_id)
            task_found = True
        else:
            # Check if task is in a merged block
            for block in self.merged_blocks:
                if task_id in block.task_ids:
                    self.current_task = task_id
                    self.current_block = block
                    task_found = True
                    
                    # Unfreeze the task head for fine-tuning
                    task_head = block.get_task_head(task_id)
                    if task_head is not None:
                        for param in task_head.parameters():
                            param.requires_grad = True
                    
                    # Freeze all other parameters
                    for param in self.parameters():
                        param.requires_grad = False
                    
                    # Only unfreeze the specific task head and block unknown head
                    if task_head is not None:
                        for param in task_head.parameters():
                            param.requires_grad = True
                    
                    if hasattr(block, 'block_unknown_head'):
                        for param in block.block_unknown_head.parameters():
                            param.requires_grad = True
                    break
        
        if not task_found:
            raise ValueError(f"Task {task_id} not found in model")

    def forward(self, x: torch.Tensor, task_id: Optional[str] = None, return_features: bool = False) -> torch.Tensor:
        """Override to handle hierarchical forward pass"""
        if task_id is None:
            task_id = self.current_task
        
        if task_id is None:
            raise ValueError("No task specified for forward pass")
        
        # Check if task is in a merged block
        for block in self.merged_blocks:
            if task_id in block.task_ids:
                # Use merged block's forward
                if return_features:
                    return block.forward_features_with_task(x, task_id, self.backbone)
                else:
                    task_index = block.task_to_index[task_id]
                    return block.forward_task(x, task_index, self.backbone)
        
        # Otherwise use parent's forward for active tasks
        return super().forward(x, task_id, return_features)
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Override to handle hierarchical structure"""
        if self.current_task is None:
            return []
        
        params = []
        
        # Check if current task is in active block (not merged)
        if self.current_task in self.active_block_tasks:
            # Get parameters from active task
            if self.current_task in self.task_adapters:
                params.extend(self.task_adapters[self.current_task].parameters())
            if self.current_task in self.task_heads:
                params.extend(self.task_heads[self.current_task].parameters())
            
            # Add block unknown head parameters if it's being trained
            if len(self.block_unknown_heads) > self.current_block_id:
                params.extend(self.block_unknown_heads[self.current_block_id].parameters())
        else:
            # Task is in a merged block - check which block
            for block in self.merged_blocks:
                if self.current_task in block.task_ids:
                    # For merged blocks, we might want to fine-tune the task head
                    # but the adapters are frozen
                    task_head = block.get_task_head(self.current_task)
                    if task_head is not None:
                        # Only train the task head for merged tasks
                        for param in task_head.parameters():
                            param.requires_grad = True
                            params.append(param)
                    
                    # Optionally, allow fine-tuning of block unknown head
                    if hasattr(block, 'block_unknown_head'):
                        for param in block.block_unknown_head.parameters():
                            param.requires_grad = True
                            params.append(param)
                    break
        
        # If still no parameters (e.g., everything is frozen), return a dummy parameter
        # to avoid optimizer error
        if not params:
            # Create a small dummy parameter that won't affect anything
            if not hasattr(self, '_dummy_param'):
                self._dummy_param = nn.Parameter(torch.zeros(1), requires_grad=True)
            params = [self._dummy_param]
        
        return params

    def load_task_checkpoint(self, task_id: str, path: str):
        """Override to handle tasks in merged blocks"""
        checkpoint = torch.load(path)
        
        # Check if task is already in a merged block
        for block in self.merged_blocks:
            if task_id in block.task_ids:
                print(f"Warning: Task {task_id} is in a merged block. Loading might not work as expected.")
                return
        
        # If not in merged block, proceed normally
        if task_id not in self.task_adapters:
            self.add_task(task_id, checkpoint['num_classes'])
        
        self.task_adapters[task_id].load_state_dict(checkpoint['adapter_state'])
        self.task_heads[task_id].load_state_dict(checkpoint['head_state'])

    def get_statistics(self) -> Dict:
        """Get hierarchical organization statistics"""
        stats = {
            'num_merged_blocks': len(self.merged_blocks),
            'tasks_in_active_block': len(self.active_block_tasks),
            'total_tasks': self.num_tasks,
            'tasks_per_block': self.tasks_per_block,
            'blocks': []
        }
        
        # Statistics for each merged block
        for block in self.merged_blocks:
            block_stats = {
                'block_id': block.block_id,
                'num_tasks': len(block.task_ids),
                'task_ids': block.task_ids
            }
            stats['blocks'].append(block_stats)
        
        # Active block info
        if self.active_block_tasks:
            stats['active_block'] = {
                'block_id': self.current_block_id,
                'task_ids': list(self.active_block_tasks.keys())
            }
        
        return stats
    
    def visualize_hierarchy(self):
        """Print a visual representation of the hierarchical structure"""
        print("\n" + "="*60)
        print("HIERARCHICAL LORA STRUCTURE")
        print("="*60)
        
        # Merged blocks
        for block in self.merged_blocks:
            print(f"\n📦 Block {block.block_id} (Merged - Orthogonal)")
            for task_id in block.task_ids:
                print(f"  └── {task_id}")
        
        # Active block
        if self.active_block_tasks:
            print(f"\n📂 Block {self.current_block_id} (Active)")
            for task_id in self.active_block_tasks.keys():
                print(f"  └── {task_id}")
        
        print("\n" + "="*60)
        
        # Summary
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")
        print(f"Memory efficiency: ~{len(self.merged_blocks)}x reduction")
        print("="*60)
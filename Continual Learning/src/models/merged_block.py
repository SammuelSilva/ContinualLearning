"""
Merged block using intelligent TIES
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from src.models.lora_vit import LoRAAttention, LoRAMlp
from src.models.lora_layers import TaskSpecificLoRA
from src.models.merge_strategies import IntelligentLoRAMerger
from src.utils.helpers import MergeConfig

class IntelligentMergedBlock(nn.Module):
    """
    Merged block using intelligent TIES
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
        merge_config: MergeConfig = None
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
        self.merge_config = merge_config or MergeConfig()
        
        # Store task information
        self.task_adapters = nn.ModuleDict(task_adapters)
        self.task_heads = nn.ModuleDict(task_heads)
        self.task_ids = list(task_adapters.keys())
        self.task_to_index = {tid: i for i, tid in enumerate(self.task_ids)}
        
        # Create intelligently merged LoRA modules
        self._create_intelligent_merged_modules()
        
        # Freeze all parameters in merged adapters
        for param in self.parameters():
            param.requires_grad = False
        
        # Create block-level unknown head
        self.block_unknown_head = nn.Linear(hidden_dim, 2)
        
        if torch.cuda.is_available():
            self.to(torch.device('cuda:0'))
    
    def _create_intelligent_merged_modules(self):
        """Create merged modules using intelligent TIES merging"""
        self.merged_attention_modules = nn.ModuleList()
        self.merged_ffn_modules = nn.ModuleList()
        
        merger = IntelligentLoRAMerger(self.merge_config)
        
        for layer_idx in range(self.num_layers):
            # Merge attention modules for this layer
            if self.lora_config in ["attention_only", "both"]:
                attn_merged = self._merge_layer_attention_intelligent(layer_idx, merger)
                self.merged_attention_modules.append(attn_merged)
            else:
                self.merged_attention_modules.append(None)
            
            # Merge FFN modules for this layer
            if self.lora_config in ["ffn_only", "both"]:
                ffn_merged = self._merge_layer_ffn_intelligent(layer_idx, merger)
                self.merged_ffn_modules.append(ffn_merged)
            else:
                self.merged_ffn_modules.append(None)
    
    def _merge_layer_attention_intelligent(self, layer_idx: int, merger):
        """Merge attention adapters for a specific layer using TIES"""
        # Collect Q and V adapters for this layer
        q_loras = {}
        v_loras = {}
        
        for task_id in self.task_ids:
            task_adapter = self.task_adapters[task_id]
            if task_adapter.attention_adapters[layer_idx] is not None:
                attn_adapter = task_adapter.attention_adapters[layer_idx]
                if 'q' in attn_adapter.lora_modules:
                    q_loras[task_id] = attn_adapter.lora_modules['q']
                if 'v' in attn_adapter.lora_modules:
                    v_loras[task_id] = attn_adapter.lora_modules['v']
        
        # Merge using TIES
        merged_q = merger.merge_group_TIES(q_loras, use_surgical=True) if q_loras else None
        merged_v = merger.merge_group_TIES(v_loras, use_surgical=True) if v_loras else None
        
        # Create merged module
        class MergedAttentionModule(nn.Module):
            def __init__(self, q_module, v_module, task_mapping):
                super().__init__()
                self.q_module = q_module
                self.v_module = v_module
                self.task_mapping = task_mapping
                self.task_q = nn.ModuleDict(task_mapping.get('q', {}))
                self.task_v = nn.ModuleDict(task_mapping.get('v', {}))
            
            def forward(self, x: torch.Tensor, task_id: Optional[str] = None) -> Dict[str, torch.Tensor]:
                outputs = {}
                if task_id and task_id in self.task_q:
                    outputs['q'] = self.task_q[task_id](x)
                    outputs['v'] = self.task_v[task_id](x)
                else:
                    if self.q_module is not None:
                        outputs['q'] = self.q_module(x)
                    if self.v_module is not None:
                        outputs['v'] = self.v_module(x)
                return outputs
        
        return MergedAttentionModule(merged_q, merged_v, {'q': q_loras, 'v': v_loras})
    
    def _merge_layer_ffn_intelligent(self, layer_idx: int, merger):
        """Merge FFN adapters for a specific layer using TIES"""
        fc1_loras = {}
        fc2_loras = {}
        
        for task_id in self.task_ids:
            task_adapter = self.task_adapters[task_id]
            if task_adapter.ffn_adapters[layer_idx] is not None:
                ffn_adapter = task_adapter.ffn_adapters[layer_idx]
                fc1_loras[task_id] = ffn_adapter.lora_fc1
                fc2_loras[task_id] = ffn_adapter.lora_fc2
        
        merged_fc1 = merger.merge_group_TIES(fc1_loras, use_surgical=True) if fc1_loras else None
        merged_fc2 = merger.merge_group_TIES(fc2_loras, use_surgical=True) if fc2_loras else None
        
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
                if task_id:
                    if fc1_input is not None and task_id in self.task_fc1:
                        outputs['fc1'] = self.task_fc1[task_id](fc1_input)
                    if fc2_input is not None and task_id in self.task_fc2:
                        outputs['fc2'] = self.task_fc2[task_id](fc2_input)
                else:
                    if fc1_input is not None and self.fc1_module is not None:
                        outputs['fc1'] = self.fc1_module(fc1_input)
                    if fc2_input is not None and self.fc2_module is not None:
                        outputs['fc2'] = self.fc2_module(fc2_input)
                return outputs
        
        return MergedFFNModule(merged_fc1, merged_fc2, {'fc1': fc1_loras, 'fc2': fc2_loras})
    
    # Rest of the methods remain the same as original...
    def inject_into_backbone(self, backbone: nn.Module, task_id: Optional[str] = None):
        """Inject merged LoRA modules into backbone"""
        for layer_idx, block in enumerate(backbone.blocks):
            original_attn = self.original_blocks[layer_idx]['attn']
            original_mlp = self.original_blocks[layer_idx]['mlp']
            
            if layer_idx < len(self.merged_attention_modules) and self.merged_attention_modules[layer_idx] is not None:
                merged_attn_module = self.merged_attention_modules[layer_idx]
                
                lora_attn = LoRAAttention(
                    original_attn=original_attn,
                    hidden_dim=self.hidden_dim,
                    lora_rank=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=0.1,
                    target_modules=["q", "v"]
                )
                
                if task_id and task_id in merged_attn_module.task_q:
                    lora_attn.lora_adapters["q"] = merged_attn_module.task_q[task_id]
                    lora_attn.lora_adapters["v"] = merged_attn_module.task_v[task_id]
                else:
                    if merged_attn_module.q_module:
                        lora_attn.lora_adapters["q"] = merged_attn_module.q_module
                    if merged_attn_module.v_module:
                        lora_attn.lora_adapters["v"] = merged_attn_module.v_module
                
                block.attn = lora_attn
            
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
                    lora_mlp.lora_fc1 = merged_ffn_module.task_fc1[task_id]
                    lora_mlp.lora_fc2 = merged_ffn_module.task_fc2[task_id]
                else:
                    if merged_ffn_module.fc1_module:
                        lora_mlp.lora_fc1 = merged_ffn_module.fc1_module
                    if merged_ffn_module.fc2_module:
                        lora_mlp.lora_fc2 = merged_ffn_module.fc2_module
                
                block.mlp = lora_mlp
    
    def remove_from_backbone(self, backbone: nn.Module):
        """Remove LoRA modules from backbone"""
        for layer_idx, block in enumerate(backbone.blocks):
            block.attn = self.original_blocks[layer_idx]['attn']
            block.mlp = self.original_blocks[layer_idx]['mlp']
    
    def compute_block_confidence(self, features: torch.Tensor) -> torch.Tensor:
        """Compute confidence that sample belongs to this block"""
        logits = self.block_unknown_head(features)
        confidence = F.softmax(logits, dim=-1)[:, 0]
        return confidence
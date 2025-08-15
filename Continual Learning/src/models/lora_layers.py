"""
LoRA (Low-Rank Adaptation) layers for efficient fine-tuning of transformer models.
Implements task-specific adapters for continuous learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import math

class LoRALayer(nn.Module):
    """
    Implements a single LoRA adapter layer.
    Decomposes weight updates into low-rank matrices: ΔW = BA
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            rank: int = 4,
            alpha: float = 1.0,
            dropout: float = 0.1
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank decomposition matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout_layer = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.merged = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LoRA layer."""
        if not self.merged:
            lora_output = self.dropout_layer(x) @ self.lora_A.T @ self.lora_B.T
            return lora_output * self.scaling
        return torch.zeros_like(x)

    def merge_weights(self, weight: nn.Parameter) -> nn.Parameter:
        """Merge LoRA weights into original weights for inference"""
        if not self.merged:
            weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True
        return weight

class MultiHeadLoRAAdapter(nn.Module):
    """
    LoRA adapter for multi-head attention layers in ViT.
    Can be applied to Q, K, V, or output projections.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.1,
        target_modules: List[str] = ["q", "v"],  # Based on hypothesis
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.target_modules = target_modules
        
        # Create LoRA adapters for specified modules
        self.lora_modules = nn.ModuleDict()
        
        if "q" in target_modules:
            self.lora_modules["q"] = LoRALayer(
                hidden_dim, hidden_dim, rank, alpha, dropout
            )
        if "k" in target_modules:
            self.lora_modules["k"] = LoRALayer(
                hidden_dim, hidden_dim, rank, alpha, dropout
            )
        if "v" in target_modules:
            self.lora_modules["v"] = LoRALayer(
                hidden_dim, hidden_dim, rank, alpha, dropout
            )
        if "o" in target_modules:
            self.lora_modules["o"] = LoRALayer(
                hidden_dim, hidden_dim, rank, alpha, dropout
            )
    
    def forward(
        self, 
        q: Optional[torch.Tensor] = None,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        o: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Apply LoRA to specified attention components"""
        outputs = {}
        
        if q is not None and "q" in self.lora_modules:
            outputs["q"] = self.lora_modules["q"](q)
        
        if k is not None and "k" in self.lora_modules:
            outputs["k"] = self.lora_modules["k"](k)
            
        if v is not None and "v" in self.lora_modules:
            outputs["v"] = self.lora_modules["v"](v)
            
        if o is not None and "o" in self.lora_modules:
            outputs["o"] = self.lora_modules["o"](o)
            
        return outputs

class FFNLoRAAdapter(nn.Module):
    """
    LoRA adapter for Feed-Forward Network layers in ViT.
    Tests the "how" hypothesis - representation learning.
    """
    def __init__(
        self,
        hidden_dim: int,
        mlp_dim: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # LoRA for both FC layers in FFN
        self.lora_fc1 = LoRALayer(hidden_dim, mlp_dim, rank, alpha, dropout)
        self.lora_fc2 = LoRALayer(mlp_dim, hidden_dim, rank, alpha, dropout)
        
    def forward(
        self,
        fc1_input: Optional[torch.Tensor] = None,
        fc2_input: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Apply LoRA to FFN layers"""
        outputs = {}
        
        if fc1_input is not None:
            outputs["fc1"] = self.lora_fc1(fc1_input)
            
        if fc2_input is not None:
            outputs["fc2"] = self.lora_fc2(fc2_input)
            
        return outputs


class TaskSpecificLoRA(nn.Module):
    """
    Complete LoRA adapter set for a specific task.
    Combines attention and/or FFN adapters based on configuration.
    """
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        mlp_dim: int,
        num_heads: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.1,
        lora_config: str = "attention_only",  # "attention_only", "ffn_only", "both"
    ):
        super().__init__()
        self.num_layers = num_layers
        self.lora_config = lora_config
        
        # Create LoRA adapters for each transformer layer
        self.attention_adapters = nn.ModuleList()
        self.ffn_adapters = nn.ModuleList()
        
        for _ in range(num_layers):
            # Attention adapters (for "what" hypothesis)
            if lora_config in ["attention_only", "both"]:
                self.attention_adapters.append(
                    MultiHeadLoRAAdapter(
                        hidden_dim, num_heads, rank, alpha, dropout,
                        target_modules=["q", "v"]  # Focus on Q and V for stability
                    )
                )
            else:
                self.attention_adapters.append(None)
            
            # FFN adapters (for "how" hypothesis)
            if lora_config in ["ffn_only", "both"]:
                self.ffn_adapters.append(
                    FFNLoRAAdapter(hidden_dim, mlp_dim, rank, alpha, dropout)
                )
            else:
                self.ffn_adapters.append(None)
    
    def get_layer_adapters(self, layer_idx: int):
        """Get adapters for a specific layer"""
        return {
            "attention": self.attention_adapters[layer_idx],
            "ffn": self.ffn_adapters[layer_idx]
        }
    
    def num_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def reset_parameters(self):
        """Reset all LoRA parameters"""
        for module in self.modules():
            if isinstance(module, LoRALayer):
                nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
                nn.init.zeros_(module.lora_B)
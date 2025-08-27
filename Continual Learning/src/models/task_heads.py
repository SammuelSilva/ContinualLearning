"""
Task-specific classification heads with unknown class mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskHead(nn.Module):
    """
    Classification head with optional unknown class for OOD detection.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        include_unknown: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.include_unknown = include_unknown
        
        # Total output dimensions
        out_features = num_classes + 1 if include_unknown else num_classes
        
        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, out_features)
        
        # Initialize
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        self.classifier.bias.data[-1] = 2.0  # Positive bias for unknown

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            self.to(device)
            
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through classification head"""
        x = self.dropout(features)
        logits = self.classifier(x)
        return logits
    
    def get_unknown_logit(self, logits: torch.Tensor) -> torch.Tensor:
        """Extract unknown class logit"""
        if self.include_unknown:
            return logits[:, -1]
        return None

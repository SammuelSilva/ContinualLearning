"""
Proper LoRA-ViT implementation with actual integration into transformer layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import timm
from timm.models.vision_transformer import VisionTransformer, Attention, Mlp
from functools import partial
from src.models.lora_layers import LoRALayer, TaskSpecificLoRA
from src.models.task_heads import TaskHead


class LoRAAttention(nn.Module):
    """
    Modified Attention module with LoRA adapters.
    Wraps the original attention module and adds LoRA to Q, K, V projections.
    """
    def __init__(
        self, 
        original_attn: Attention,
        hidden_dim: int,
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.1,
        target_modules: List[str] = ["q", "v"]
    ):
        super().__init__()

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            self.to(device)

        self.original_attn = original_attn
        self.hidden_dim = hidden_dim
        
        # Create LoRA adapters for specified targets
        self.lora_adapters = nn.ModuleDict()
        
        if "q" in target_modules:
            self.lora_adapters["q"] = LoRALayer(
                hidden_dim, hidden_dim, lora_rank, lora_alpha, lora_dropout
            )
        if "k" in target_modules:
            self.lora_adapters["k"] = LoRALayer(
                hidden_dim, hidden_dim, lora_rank, lora_alpha, lora_dropout
            )
        if "v" in target_modules:
            self.lora_adapters["v"] = LoRALayer(
                hidden_dim, hidden_dim, lora_rank, lora_alpha, lora_dropout
            )
            
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        
        print(f"\n[DEBUG LoRAAttention.forward] Input shape: {x.shape}")
        print(f"  - self.original_attn type: {type(self.original_attn)}")
        print(f"  - self.original_attn.qkv type: {type(self.original_attn.qkv)}")

        # Get Q, K, V from original attention
        self.qkv = self.original_attn.qkv(x).reshape(B, N, 3, self.original_attn.num_heads, C // self.original_attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = self.qkv.unbind(0)  # B, num_heads, N, head_dim
        
        # Apply LoRA adapters if present
        if "q" in self.lora_adapters:
            q_adapt = self.lora_adapters["q"](x)  # B, N, C
            q_adapt = q_adapt.reshape(B, N, self.original_attn.num_heads, C // self.original_attn.num_heads).permute(0, 2, 1, 3)
            q = q + q_adapt
            
        if "k" in self.lora_adapters:
            k_adapt = self.lora_adapters["k"](x)
            k_adapt = k_adapt.reshape(B, N, self.original_attn.num_heads, C // self.original_attn.num_heads).permute(0, 2, 1, 3)
            k = k + k_adapt
            
        if "v" in self.lora_adapters:
            v_adapt = self.lora_adapters["v"](x)
            v_adapt = v_adapt.reshape(B, N, self.original_attn.num_heads, C // self.original_attn.num_heads).permute(0, 2, 1, 3)
            v = v + v_adapt
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.original_attn.scale
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn = attn + attn_mask
            
        attn = attn.softmax(dim=-1)
        attn = self.original_attn.attn_drop(attn)
        
        # Apply to values and project
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.original_attn.proj(x)
        x = self.original_attn.proj_drop(x)
        
        return x


class LoRAMlp(nn.Module):
    """
    Modified MLP module with LoRA adapters for FFN layers.
    """
    def __init__(
        self,
        original_mlp: Mlp,
        hidden_dim: int,
        mlp_dim: int,
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.1
    ):
        super().__init__()

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            self.to(device)

        self.original_mlp = original_mlp
        
        # LoRA for FC layers
        self.lora_fc1 = LoRALayer(
            hidden_dim, mlp_dim, lora_rank, lora_alpha, lora_dropout
        )
        self.lora_fc2 = LoRALayer(
            mlp_dim, hidden_dim, lora_rank, lora_alpha, lora_dropout
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First FC layer with LoRA
        h = self.original_mlp.fc1(x)
        h = h + self.lora_fc1(x)
        h = self.original_mlp.act(h)
        h = self.original_mlp.drop1(h)
        
        # Second FC layer with LoRA
        h_pre = h
        h = self.original_mlp.fc2(h)
        h = h + self.lora_fc2(h_pre)
        h = self.original_mlp.drop2(h)
        
        return h

class ContinualLoRAViT(nn.Module):
    """
    Proper implementation of Vision Transformer with task-specific LoRA adapters.
    This version correctly integrates LoRA into the transformer layers.
    """
    
    def __init__(
        self,
        vit_model_name: str = "vit_base_patch16_224",
        num_classes_per_task: List[int] = None,
        lora_rank: int = 4,
        lora_alpha: float = 4.0,
        lora_dropout: float = 0.1,
        lora_config: str = "attention_only",
        use_pretrained: bool = True,
    ):
        super().__init__()
        
        # Load pre-trained ViT
        self.backbone = timm.create_model(
            vit_model_name,
            pretrained=use_pretrained,
            num_classes=0  # Remove classification head
        )
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Extract model dimensions from backbone
        self.hidden_dim = self.backbone.embed_dim
        self.num_heads = self.backbone.blocks[0].attn.num_heads
        self.num_layers = len(self.backbone.blocks)
        
        # Determine MLP dimension
        self.mlp_dim = self.backbone.blocks[0].mlp.fc1.out_features
        
        # Configuration
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_config = lora_config
        
        # Task management
        self.task_adapters = nn.ModuleDict()
        self.task_heads = nn.ModuleDict()
        self.num_tasks = 0
        self.current_task = None
        self.task_classes = {}
        
        # Store original modules for restoration
        self._store_original_modules()
        
        # Initialize tasks if provided
        if num_classes_per_task:
            for task_id, num_classes in enumerate(num_classes_per_task):
                self.add_task(f"task_{task_id}", num_classes)
    
    def _store_original_modules(self):
        """Store references to original attention and MLP modules"""
        self.original_blocks = []
        for block in self.backbone.blocks:
            self.original_blocks.append({
                'attn': block.attn,
                'mlp': block.mlp
            })
    
    def _inject_lora_modules(self, task_id: str):
        """
        Inject LoRA modules for a specific task into the transformer blocks.
        This is the key function that actually modifies the forward pass.
        """
        task_lora = self.task_adapters[task_id]
        
        for idx, block in enumerate(self.backbone.blocks):
            # Get LoRA adapters for this layer
            layer_adapters = task_lora.get_layer_adapters(idx)
            
            # Replace attention module with LoRA-enhanced version
            if layer_adapters['attention'] is not None and self.lora_config in ["attention_only", "both"]:
                # Create new attention module with LoRA
                lora_attn = LoRAAttention(
                    original_attn=self.original_blocks[idx]['attn'],
                    hidden_dim=self.hidden_dim,
                    lora_rank=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                    target_modules=["q", "v"]  # Based on hypothesis
                )
                # Copy LoRA weights from task adapter
                lora_attn.lora_adapters = layer_adapters['attention'].lora_modules
                block.attn = lora_attn
            
            # Replace MLP module with LoRA-enhanced version
            if layer_adapters['ffn'] is not None and self.lora_config in ["ffn_only", "both"]:
                # Create new MLP module with LoRA
                lora_mlp = LoRAMlp(
                    original_mlp=self.original_blocks[idx]['mlp'],
                    hidden_dim=self.hidden_dim,
                    mlp_dim=self.mlp_dim,
                    lora_rank=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout
                )
                # Copy LoRA weights from task adapter
                lora_mlp.lora_fc1 = layer_adapters['ffn'].lora_fc1
                lora_mlp.lora_fc2 = layer_adapters['ffn'].lora_fc2
                block.mlp = lora_mlp
    
    def _restore_original_modules(self):
        """Restore original modules (remove LoRA)"""
        for idx, block in enumerate(self.backbone.blocks):
            block.attn = self.original_blocks[idx]['attn']
            block.mlp = self.original_blocks[idx]['mlp']
    
    def add_task(self, task_id: str, num_classes: int):
        """Add a new task with its LoRA adapters and classification head"""
        
        # Create task-specific LoRA adapters
        self.task_adapters[task_id] = TaskSpecificLoRA(
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
            lora_config=self.lora_config,
        )
        
        # Create task-specific head with unknown class
        self.task_heads[task_id] = TaskHead(
            hidden_dim=self.hidden_dim,
            num_classes=num_classes,
            include_unknown=True
        )
        
        self.task_classes[task_id] = num_classes
        self.num_tasks += 1
        
        print(f"Added task '{task_id}' with {num_classes} classes (+1 unknown)")
        print(f"LoRA parameters: {self.task_adapters[task_id].num_parameters():,}")
    
    def set_active_task(self, task_id: str):
        """Set the current active task and inject its LoRA modules"""
        if task_id not in self.task_adapters:
            raise ValueError(f"Task {task_id} not found")
        
        # Restore original modules first
        self._restore_original_modules()
        
        # Inject LoRA modules for the new task
        self._inject_lora_modules(task_id)
        
        self.current_task = task_id
        
        # Set gradients appropriately
        for param in self.parameters():
            param.requires_grad = False
        
        # Enable gradients only for current task
        for param in self.task_adapters[task_id].parameters():
            param.requires_grad = True
        for param in self.task_heads[task_id].parameters():
            param.requires_grad = True
    
    def forward_features_with_lora(
        self, 
        x: torch.Tensor, 
        task_id: Optional[str] = None
    ) -> torch.Tensor:
        """
        Forward pass through backbone with task-specific LoRA adapters.
        This properly applies LoRA by injecting modules before forward pass.
        """
        if task_id is None:
            task_id = self.current_task
        
        if task_id is None:
            raise ValueError("No task specified for forward pass")
        
        # Ensure correct LoRA modules are injected
        if self.current_task != task_id:
            self._restore_original_modules()
            self._inject_lora_modules(task_id)
            self.current_task = task_id
        
        # Now forward pass will use LoRA-enhanced modules
        features = self.backbone(x)
        
        return features
    
    def forward(
        self,
        x: torch.Tensor,
        task_id: Optional[str] = None,
        return_features: bool = False
    ) -> torch.Tensor:
        """Forward pass through model with LoRA adaptation"""
        
        # Get features with LoRA adaptation
        features = self.forward_features_with_lora(x, task_id)
        
        if return_features:
            return features
        
        # Apply task-specific head
        if task_id is None:
            task_id = self.current_task
        
        logits = self.task_heads[task_id](features)
        return logits
    
    def predict_task_id(self, x: torch.Tensor) -> Tuple[List[str], torch.Tensor]:
        """
        Predict task ID using unknown class probabilities.
        Returns task with lowest unknown class probability.
        """
        batch_size = x.shape[0]
        unknown_probs = {}
        all_logits = {}
        
        with torch.no_grad():
            for task_id in self.task_adapters.keys():
                # Forward through task-specific model (with proper LoRA injection)
                logits = self.forward(x, task_id=task_id)
                all_logits[task_id] = logits
                
                # Get probability of unknown class (last class)
                probs = F.softmax(logits, dim=-1)
                unknown_probs[task_id] = probs[:, -1]  # Last index is unknown
        
        # Stack unknown probabilities
        unknown_prob_matrix = torch.stack(list(unknown_probs.values()), dim=1)
        
        # Find task with minimum unknown probability for each sample
        min_unknown_probs, task_indices = torch.min(unknown_prob_matrix, dim=1)
        
        # Map indices to task IDs
        task_id_list = list(self.task_adapters.keys())
        predicted_tasks = [task_id_list[idx] for idx in task_indices]
        
        return predicted_tasks, min_unknown_probs
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """
        Full prediction pipeline:
        1. Predict task ID using unknown class
        2. Predict class within identified task
        """
        # Predict task IDs
        predicted_tasks, unknown_probs = self.predict_task_id(x)
        
        # Collect predictions for each sample
        batch_size = x.shape[0]
        predictions = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        
        with torch.no_grad():
            for i in range(batch_size):
                task_id = predicted_tasks[i]
                
                # Get logits for this task (excluding unknown class)
                logits = self.forward(x[i:i+1], task_id=task_id)
                logits_without_unknown = logits[:, :-1]
                
                # Predict class within task
                pred = torch.argmax(logits_without_unknown, dim=1)
                predictions[i] = pred
        
        return predictions, predicted_tasks
    
    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: str,
        memory_buffer: Optional[Dict] = None,
        lambda_unknown: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for current task training.
        Includes unknown class loss for memory buffer samples.
        """
        losses = {}
        
        # Forward pass with proper LoRA injection
        logits = self.forward(x, task_id=task_id)
        
        # Current task loss (excluding unknown class)
        logits_task = logits[:, :-1]
        loss_current = F.cross_entropy(logits_task, y)
        losses['current_task'] = loss_current
        
        # Memory buffer loss (samples labeled as unknown)
        if memory_buffer is not None and len(memory_buffer['images']) > 0:
            mem_images = memory_buffer['images']
            mem_logits = self.forward(mem_images, task_id=task_id)
            
            # Label all memory samples as unknown (last class)
            unknown_labels = torch.full(
                (len(mem_images),),
                self.task_classes[task_id],  # Unknown class index
                dtype=torch.long,
                device=mem_logits.device
            )
            
            loss_unknown = F.cross_entropy(mem_logits, unknown_labels)
            losses['unknown'] = loss_unknown
            
            # Combined loss
            losses['total'] = loss_current + lambda_unknown * loss_unknown
        else:
            losses['total'] = loss_current
        
        return losses
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get all trainable parameters for current task"""
        if self.current_task is None:
            return []
        
        params = []
        params.extend(self.task_adapters[self.current_task].parameters())
        params.extend(self.task_heads[self.current_task].parameters())
        return params
    
    def save_task_checkpoint(self, task_id: str, path: str):
        checkpoint = {
            'task_id': task_id,
            'adapter_state': self.task_adapters[task_id].state_dict(),
            'head_state': self.task_heads[task_id].state_dict(),
            'num_classes': self.task_classes[task_id],
            'lora_config': self.lora_config,
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha
        }
        torch.save(checkpoint, path)

    def load_task_checkpoint(self, task_id: str, path: str):
        checkpoint = torch.load(path)

        if task_id not in self.task_adapters:
            self.add_task(task_id, checkpoint['num_classes'])
        
        self.task_adapters[task_id].load_state_dict(checkpoint['adapter_state'])
        self.task_heads[task_id].load_state_dict(checkpoint['head_state'])
    
    def freeze_previous_tasks(self):
        for task_id in self.task_adapters.keys():
            if task_id != self.current_task:
                for param in self.task_adapters[task_id].parameters():
                    param.requires_grad = False
                for param in self.task_heads[task_id].parameters():
                    param.requires_grad = False


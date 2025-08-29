"""
Hierarchical LoRA management with orthogonal merging capability.
Extends the existing ContinualLoRAViT with block-based organization.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from src.models.lora_vit import ContinualLoRAViT, LoRAAttention, LoRAMlp
from src.models.lora_layers import TaskSpecificLoRA
from src.models.merged_block import IntelligentMergedBlock
from src.utils.helpers import MergeConfig
from src.models.merge_strategies import IntelligentLoRAMerger

class HierarchicalLoRAViT(ContinualLoRAViT):
    """
    Hierarchical LoRA-ViT with intelligent TIES merging
    Modified to use memory buffer for merge validation
    """
    
    def __init__(
        self,
        vit_model_name: str = "vit_base_patch16_224",
        lora_rank: int = 4,
        lora_alpha: float = 4.0,
        lora_dropout: float = 0.1,
        lora_config: str = "attention_only",
        use_pretrained: bool = True,
        max_tasks_per_block: int = 5,
        min_tasks_to_merge: int = 2,
        merge_config: MergeConfig = None,
        memory_buffer=None
    ):
        super().__init__(
            vit_model_name=vit_model_name,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_config=lora_config,
            use_pretrained=use_pretrained
        )
        
        # Task management
        self.specialist_tasks = {}
        self.merged_blocks = nn.ModuleList()
        
        # Merge configuration
        self.max_tasks_per_block = max_tasks_per_block
        self.min_tasks_to_merge = min_tasks_to_merge
        self.merge_config = merge_config or MergeConfig()
        
        # Memory buffer reference
        self.memory_buffer = memory_buffer
        
        # Statistics
        self.total_tasks = 0
        self.merge_attempts = 0
        self.successful_merges = 0
        
        if torch.cuda.is_available():
            self.to(torch.device('cuda:0'))
    
    def set_memory_buffer(self, memory_buffer):
        """Set or update the memory buffer reference"""
        self.memory_buffer = memory_buffer
    
    def add_task(self, task_id: str, num_classes: int):
        """
        Add a new task as a specialist
        No longer needs validation_data parameter
        """
        # Create task components using parent's method
        super().add_task(task_id, num_classes)
        
        # Store as specialist
        self.specialist_tasks[task_id] = {
            'adapter': self.task_adapters[task_id],
            'head': self.task_heads[task_id],
            'num_classes': num_classes,
            'trained': False
        }
        
        self.total_tasks += 1
        
        print(f"Added task '{task_id}' as specialist (Total: {self.total_tasks}, "
              f"Unmerged: {len(self.specialist_tasks)}, Blocks: {len(self.merged_blocks)})")
        
        # Check if we should attempt merging
        unmerged_count = len(self.specialist_tasks)
        if unmerged_count >= self.max_tasks_per_block:
            print(f"\n📦 Reached {self.max_tasks_per_block} unmerged tasks. Attempting intelligent merge...")
            self._attempt_intelligent_merging()
    
    def _attempt_intelligent_merging(self):
        """
        Attempt to merge compatible specialists using memory buffer for validation
        """
        from src.models.merge_strategies import IntelligentLoRAMerger
        
        self.merge_attempts += 1
        merger = IntelligentLoRAMerger(self.merge_config)
        
        # Get all unmerged specialists
        available_specialists = {
            tid: self.task_adapters[tid] 
            for tid in self.specialist_tasks.keys()
            if tid in self.task_adapters
        }
        
        if len(available_specialists) < self.min_tasks_to_merge:
            print(f"Not enough specialists to merge ({len(available_specialists)} < {self.min_tasks_to_merge})")
            return
        
        # Phase 1: Group compatible tasks
        groups = merger.group_adapters(available_specialists)
        print(f"Found {len(groups)} potential groups from {len(available_specialists)} specialists")
        
        merged_any = False
        
        for i, group in enumerate(groups):
            # Skip groups that are too small
            if len(group) < self.min_tasks_to_merge:
                print(f"  Group {i+1}: {group} - Too small to merge")
                continue
            
            # If group exceeds max size, take the most similar subset
            if len(group) > self.max_tasks_per_block:
                group = group[:self.max_tasks_per_block]
                print(f"  Group {i+1}: Trimmed to {group} (max size: {self.max_tasks_per_block})")
            else:
                print(f"  Group {i+1}: {group}")
            
            # Try to merge this group with buffer validation
            success = self._try_merge_group(group, merger)
            if success:
                merged_any = True
                self.successful_merges += 1
        
        if not merged_any:
            print("⚠️ No successful merges. All tasks remain as specialists.")
        
        # Print current status
        self._print_status()
    
    def _try_merge_group(self, group: List[str], merger) -> bool:
        """
        Attempt to merge a specific group using memory buffer for validation
        """
        print(f"  Attempting to merge: {group}")
        
        # Collect adapters and heads
        group_adapters = {}
        group_heads = {}
        for task_id in group:
            if task_id in self.task_adapters:
                group_adapters[task_id] = self.task_adapters[task_id]
                group_heads[task_id] = self.task_heads[task_id]
        
        if len(group_adapters) < len(group):
            print(f"  ❌ Some tasks in group not found")
            return False
        
        # Phase 2: Merge using TIES with surgical promotion
        try:
            merged_lora = merger.merge_group_TIES(group_adapters, use_surgical=True)
        except Exception as e:
            print(f"  ❌ Merge failed: {str(e)}")
            return False
        
        # Phase 3: Validate using memory buffer
        if self.memory_buffer and len(self.memory_buffer) > 0:
            print(f"    Validating with memory buffer ({len(self.memory_buffer)} samples)...")
            validation_passed = merger.validate_merge_with_buffer(
                group_adapters,
                merged_lora,
                self.memory_buffer,
                self.backbone,
                group_heads
            )
        else:
            print(f"    Warning: No memory buffer available, accepting merge without validation")
            validation_passed = True
        
        if validation_passed:
            print(f"  ✅ Validation passed. Creating merged block.")
            
            # Create merged block
            from src.models.hierarchical_lora import IntelligentMergedBlock
            
            block = IntelligentMergedBlock(
                original_blocks=self.original_blocks,
                task_adapters=group_adapters,
                task_heads=group_heads,
                block_id=len(self.merged_blocks),
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_config=self.lora_config,
                merge_config=self.merge_config
            )
            
            self.merged_blocks.append(block)
            
            # Remove merged tasks from specialists
            for task_id in group:
                if task_id in self.task_adapters:
                    del self.task_adapters[task_id]
                if task_id in self.task_heads:
                    del self.task_heads[task_id]
                if task_id in self.specialist_tasks:
                    del self.specialist_tasks[task_id]
            
            return True
        else:
            print(f"  ❌ Validation failed. Keeping tasks as specialists.")
            return False
    
    def forward(self, x: torch.Tensor, task_id: Optional[str] = None, 
                return_features: bool = False) -> torch.Tensor:
        """
        Forward pass for a specific task
        Task can be either a specialist or in a merged block
        """
        if task_id is None:
            task_id = self.current_task
        
        if task_id is None:
            raise ValueError("No task specified for forward pass")
        
        # Check if task is in a merged block
        for block in self.merged_blocks:
            if task_id in block.task_ids:
                features = self._forward_with_block(x, block, task_id)
                if return_features:
                    return features
                return block.task_heads[task_id](features)
        
        # Otherwise, use parent's forward for specialist tasks
        return super().forward(x, task_id, return_features)
    
    def _forward_with_block(self, x: torch.Tensor, block, task_id: str) -> torch.Tensor:
        """Forward pass through backbone with merged block's LoRA"""
        # Inject block's adapters
        block.inject_into_backbone(self.backbone, task_id)
        
        try:
            # Forward through modified backbone
            features = self.backbone(x)
        finally:
            # Always restore
            block.remove_from_backbone(self.backbone)
        
        return features
    
    def predict_task_id(self, x: torch.Tensor) -> Tuple[List[str], torch.Tensor]:
        """
        Predict task ID using unknown class probabilities
        Checks both specialists and merged blocks
        """
        batch_size = x.shape[0]
        all_candidates = {}
        
        with torch.no_grad():
            # Get backbone features once
            features = self.backbone(x)
            
            # Check all specialist tasks
            for task_id in self.specialist_tasks:
                if task_id in self.task_heads:
                    logits = self.task_heads[task_id](features)
                    probs = F.softmax(logits, dim=-1)
                    unknown_probs = probs[:, -1]  # Last class is unknown
                    all_candidates[task_id] = unknown_probs
            
            # Check all tasks in merged blocks
            for block in self.merged_blocks:
                # First check block confidence
                block_confidence = block.compute_block_confidence(features)
                
                for task_id in block.task_ids:
                    logits = block.task_heads[task_id](features)
                    probs = F.softmax(logits, dim=-1)
                    unknown_probs = probs[:, -1]
                    
                    # Weight by block confidence
                    weighted_unknown = unknown_probs * (2 - block_confidence)
                    all_candidates[task_id] = weighted_unknown
        
        # Select task with lowest unknown probability for each sample
        predicted_tasks = []
        confidences = []
        
        for i in range(batch_size):
            min_unknown = float('inf')
            best_task = None
            
            for task_id, unknown_probs in all_candidates.items():
                if unknown_probs[i] < min_unknown:
                    min_unknown = unknown_probs[i]
                    best_task = task_id
            
            predicted_tasks.append(best_task if best_task else "task_0")
            confidences.append(1 - min_unknown if best_task else 0.0)
        
        return predicted_tasks, torch.tensor(confidences, device=x.device)
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get trainable parameters for current task"""
        if self.current_task is None:
            return []
        
        params = []
        
        # Check if current task is a specialist
        if self.current_task in self.task_adapters:
            params.extend(self.task_adapters[self.current_task].parameters())
            params.extend(self.task_heads[self.current_task].parameters())
        else:
            # Task is in a merged block - only train head
            for block in self.merged_blocks:
                if self.current_task in block.task_ids:
                    params.extend(block.task_heads[self.current_task].parameters())
                    params.extend(block.block_unknown_head.parameters())
                    break
        
        return params
    
    def _task_in_blocks(self, task_id: str) -> bool:
        """Check if a task is in any merged block"""
        for block in self.merged_blocks:
            if task_id in block.task_ids:
                return True
        return False
    
    def _print_status(self):
        """Print current model status"""
        print("\n" + "="*60)
        print("HIERARCHICAL MODEL STATUS")
        print("="*60)
        print(f"Total tasks: {self.total_tasks}")
        print(f"Specialist tasks (unmerged): {len(self.specialist_tasks)}")
        print(f"Merged blocks: {len(self.merged_blocks)}")
        
        if self.merged_blocks:
            print("\nMerged Blocks:")
            for i, block in enumerate(self.merged_blocks):
                print(f"  Block {i}: {block.task_ids} ({len(block.task_ids)} tasks)")
        
        if self.specialist_tasks:
            print(f"\nSpecialist Tasks: {list(self.specialist_tasks.keys())}")
        
        print(f"\nMerge Statistics:")
        print(f"  Attempts: {self.merge_attempts}")
        print(f"  Successful: {self.successful_merges}")
        if self.merge_attempts > 0:
            print(f"  Success rate: {100*self.successful_merges/self.merge_attempts:.1f}%")
        print("="*60)
    
    def get_statistics(self) -> Dict:
        """Get model statistics"""
        stats = {
            'total_tasks': self.total_tasks,
            'specialist_tasks': len(self.specialist_tasks),
            'num_merged_blocks': len(self.merged_blocks),
            'merge_attempts': self.merge_attempts,
            'successful_merges': self.successful_merges,
            'blocks': []
        }
        
        for i, block in enumerate(self.merged_blocks):
            stats['blocks'].append({
                'block_id': i,
                'num_tasks': len(block.task_ids),
                'task_ids': block.task_ids
            })
        
        if self.specialist_tasks:
            stats['specialists'] = list(self.specialist_tasks.keys())
        
        return stats
    
    def save_checkpoint(self, path: str):
        """Save complete model checkpoint"""
        checkpoint = {
            'model_state': self.state_dict(),
            'statistics': self.get_statistics(),
            'merge_config': self.merge_config,
            'specialist_tasks': self.specialist_tasks,
            'task_validation_data': self.task_validation_data
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state'])
        self.specialist_tasks = checkpoint['specialist_tasks']
        self.task_validation_data = checkpoint['task_validation_data']
        print(f"Checkpoint loaded from {path}")
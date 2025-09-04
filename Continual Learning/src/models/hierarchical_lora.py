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
        memory_buffer=None,
        gradient_checkpointing: bool = True, 
        checkpoint_segments: int = 4 
    ):
        super().__init__(
            vit_model_name=vit_model_name,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_config=lora_config,
            use_pretrained=use_pretrained,
            gradient_checkpointing=gradient_checkpointing,
            checkpoint_segments=checkpoint_segments
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
            if tid in self.task_adapters and self.specialist_tasks[tid]['trained']
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
    
    def _get_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from backbone (non-checkpointed version for inference)
        """
        x = self.backbone.patch_embed(images)
        
        # Add class token if the model expects it
        if hasattr(self.backbone, 'cls_token'):
            cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            
        # Position embeddings
        if hasattr(self.backbone, 'pos_drop'):
            x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        
        # Transformer blocks (no checkpointing for inference)
        for block in self.backbone.blocks:
            x = block(x)
        
        x = self.backbone.norm(x)
        
        # Extract class token (first token) for classification
        if hasattr(self.backbone, 'cls_token'):
            cls_output = x[:, 0]  # Shape: [batch_size, embed_dim]
        else:
            cls_output = x.mean(dim=1)  # Global average pooling
        
        # Apply backbone head if it exists
        if hasattr(self.backbone, 'head') and self.backbone.head is not None:
            features = self.backbone.head(cls_output)
        else:
            features = cls_output
        
        return features
    
    def _get_features_cpu(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features on CPU to save GPU memory
        Used for validation and buffer operations
        """
        # Move model to CPU temporarily if needed
        original_device = next(self.parameters()).device
        
        if original_device.type == 'cuda':
            self.cpu()
            images = images.cpu()
        
        # Extract features on CPU
        with torch.no_grad():
            x = self.backbone.patch_embed(images)
            
            # Add class token if the model expects it
            if hasattr(self.backbone, 'cls_token'):
                cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls_token, x), dim=1)
                
            # Position embeddings
            if hasattr(self.backbone, 'pos_drop'):
                x = self.backbone.pos_drop(x + self.backbone.pos_embed)
            
            # Transformer blocks (no checkpointing for inference)
            for block in self.backbone.blocks:
                x = block(x)
            
            x = self.backbone.norm(x)
            
            # Extract class token (first token) for classification
            if hasattr(self.backbone, 'cls_token'):
                cls_output = x[:, 0]
            else:
                cls_output = x.mean(dim=1)
            
            # Apply backbone head if it exists
            if hasattr(self.backbone, 'head') and self.backbone.head is not None:
                features = self.backbone.head(cls_output)
            else:
                features = cls_output
        
        # Move model back to original device if needed
        if original_device.type == 'cuda':
            self.to(original_device)
            # Keep features on CPU
        
        return features  # Returns CPU tensor

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
            if self.training and self.gradient_checkpointing:
                features = self._forward_with_checkpointing(x)
            else:
                features = self.backbone(x)
        finally:
            # Always restore
            block.remove_from_backbone(self.backbone)
        
        return features
    
    def set_gradient_checkpointing(self, enable: bool):
        """
        Enable or disable gradient checkpointing
        """
        self.gradient_checkpointing = enable
        
        # Try to set in backbone if supported
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(enable)
        
        print(f"Gradient checkpointing: {'ENABLED' if enable else 'DISABLED'}")

    def predict_task_id(self, x: torch.Tensor, unknown_threshold: float = 0.5, use_cpu: bool = True) -> Tuple[List[str], List[float]]:
        """
        Modified version with CPU option for memory efficiency
        """
        self.eval()
        batch_size = x.size(0)
        
        with torch.no_grad():
            # Get features on CPU if requested
            if use_cpu:
                features = self._get_features_cpu(x.cpu())
            else:
                features = self._get_features(x)
            
            head_scores = {}
            
            # Process heads on CPU
            for task_id in self.specialist_tasks:
                if task_id in self.task_heads:
                    # Move head to CPU temporarily
                    head = self.task_heads[task_id]
                    original_device = next(head.parameters()).device
                    
                    if use_cpu and original_device.type == 'cuda':
                        head.cpu()
                    
                    # Compute logits on CPU
                    logits = head(features)
                    probs = F.softmax(logits, dim=1)
                    
                    # Score calculation
                    class_probs = probs[:, :-1]
                    unknown_prob = probs[:, -1]
                    max_class_prob = torch.max(class_probs, dim=1)[0]
                    acceptance_score = max_class_prob * (1.0 - unknown_prob) 
                    head_scores[task_id] = acceptance_score
                    
                    # Move head back if needed
                    if use_cpu and original_device.type == 'cuda':
                        head.to(original_device)
            
            # Route based on highest acceptance score
            predicted_tasks = []
            confidences = []
            all_values_heads = []
            for i in range(batch_size):
                best_task = None
                best_score = -1.0
                all_task_scores = {}
                for task_id, scores in head_scores.items():
                    score = scores[i].item()
                    if score > best_score:
                        best_score = score
                        best_task = task_id
                    if all_task_scores.get(task_id):
                        all_task_scores[task_id].update({i: score})
                    else:
                        all_task_scores[task_id] = {i: score}

                predicted_tasks.append(best_task)
                confidences.append(best_score)
                all_values_heads.append(all_task_scores)
        
        return predicted_tasks, confidences, all_values_heads

    def debug_unknown_class_routing(self, x: torch.Tensor, true_task_ids: List[str] = None):
        """
        Debug the unknown class-based routing mechanism
        """
        print("\n=== Unknown Class Routing Debug ===")
        
        self.eval()
        with torch.no_grad():
            features = self._get_features(x[:4])  # First 4 samples
            
            print(f"Testing {features.shape[0]} samples...")
            
            # Test each head
            for task_id in sorted(self.specialist_tasks):
                if task_id in self.task_heads:
                    head = self.task_heads[task_id]
                    logits = head(features)
                    probs = F.softmax(logits, dim=1)
                    
                    print(f"\nHead {task_id}:")
                    for i in range(features.shape[0]):
                        unknown_prob = probs[i, -1].item()
                        class_probs = probs[i, :-1]
                        max_class_prob = torch.max(class_probs).item()
                        predicted_class = torch.argmax(class_probs).item()
                        
                        verdict = "ACCEPT" if unknown_prob < 0.3 else "REJECT"
                        print(f"  Sample {i}: Unknown={unknown_prob:.3f}, MaxClass={max_class_prob:.3f} (class {predicted_class}) -> {verdict}")
            
            # Show actual routing decisions
            pred_tasks, confs = self.predict_task_id(x[:4])
            print(f"\nRouting Decisions:")
            for i in range(4):
                true_task = true_task_ids[i] if true_task_ids else "?"
                print(f"  Sample {i}: {true_task} -> {pred_tasks[i]} (conf: {confs[i]:.3f})")
    
    def validate_unknown_class_training(self, memory_buffer):
        """
        Check if unknown class training worked properly
        """
        print("\n=== Unknown Class Training Validation ===")
        
        if len(memory_buffer) == 0:
            return
        
        # Sample data from buffer
        memory_batch = memory_buffer.sample(batch_size=32)
        mem_images = memory_batch['images'].to(self.device)
        mem_task_ids = memory_batch.get('task_ids', [])
        
        self.eval()
        with torch.no_grad():
            features = self._get_features(mem_images)
            
            # Test each head's ability to reject other tasks' data
            for task_id in self.specialist_tasks:
                if task_id in self.task_heads:
                    head = self.task_heads[task_id]
                    logits = head(features)
                    probs = F.softmax(logits, dim=1)
                    
                    # Separate own vs other task samples
                    own_indices = [i for i, tid in enumerate(mem_task_ids) if tid == task_id]
                    other_indices = [i for i, tid in enumerate(mem_task_ids) if tid != task_id]
                    
                    if own_indices and other_indices:
                        # Check unknown probabilities
                        own_unknown = probs[own_indices, -1].mean().item()
                        other_unknown = probs[other_indices, -1].mean().item()
                        
                        print(f"Head {task_id}:")
                        print(f"  Own task unknown prob: {own_unknown:.3f} (should be LOW)")
                        print(f"  Other task unknown prob: {other_unknown:.3f} (should be HIGH)")
                        
                        if own_unknown > 0.5:
                            print(f"  ⚠️ WARNING: Head rejects its own task data!")
                        if other_unknown < 0.3:
                            print(f"  ⚠️ WARNING: Head accepts other tasks' data!")
                            
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
    
    def mark_task_trained(self, task_id: str):
        """Mark a task as trained after training completion"""
        if task_id in self.specialist_tasks:
            self.specialist_tasks[task_id]['trained'] = True
    
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
    
    def visualize_hierarchy(self):
        print("\n" + "="*60)
        print("HIERARCHICAL STRUCTURE")
        print("\n" + "="*60)

        for i, block in enumerate(self.merged_blocks):
            print(f"\nBlock {i} (merged)")
            for task_id in block.task_ids:
                print(f" |___ task_id[{task_id}]")
        
        if self.specialist_tasks:
            print(f"\nSpecialists (unmerged)")
            for task_id in self.specialist_tasks.keys():
                print(f" |___ task_id[{task_id}]")

        print("\n" + "="*60)
        
    def save_checkpoint(self, path: str):
        """Save complete model checkpoint"""
        checkpoint = {
            'model_state': self.state_dict(),
            'statistics': self.get_statistics(),
            'merge_config': self.merge_config,
            'specialist_tasks': self.specialist_tasks,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state'])
        self.specialist_tasks = checkpoint['specialist_tasks']
        print(f"Checkpoint loaded from {path}")
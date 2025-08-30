"""
Advanced merging strategies for hierarchical LoRA.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dataclasses import dataclass
from typing import Dict, List, Optional
from src.utils.helpers import MergeConfig

class IntelligentLoRAMerger:
    """
    Merger that validates using memory buffer data
    """
    
    def __init__(self, config: MergeConfig = None):
        self.config = config or MergeConfig()
    
    @dataclass
    class PerformanceMetrics:
        accuracy: float
        rejection_rate: float
        fp_rate: float
    
    def validate_merge_with_buffer(self,
                                  original_loras: Dict[str, nn.Module],
                                  merged_lora: nn.Module,
                                  memory_buffer,
                                  backbone: nn.Module,
                                  task_heads: Dict[str, nn.Module]) -> bool:
        """
        Validate merge using memory buffer data
        """
        if memory_buffer is None or len(memory_buffer) == 0:
            print("  Warning: No memory buffer for validation, accepting merge")
            return True
        
        all_valid = True
        
        for task_id, original_lora in original_loras.items():
            print(f"    Validating task {task_id}...")
            
            # Get task data from memory buffer
            task_data = self._get_task_data_from_buffer(
                memory_buffer, task_id, self.config.validation_samples_per_task
            )
            ood_data = self._get_ood_data_from_buffer(
                memory_buffer, task_id, self.config.validation_samples_per_task
            )
            
            if task_data is None:
                print(f"      No data for {task_id} in buffer, skipping validation")
                continue
            
            # Evaluate original
            original_metrics = self._calculate_metrics_from_buffer(
                original_lora,
                task_data,
                ood_data,
                backbone,
                task_heads[task_id]
            )
            
            # Evaluate merged
            merged_metrics = self._calculate_metrics_from_buffer(
                merged_lora,
                task_data,
                ood_data,
                backbone,
                task_heads[task_id]
            )
            
            # Check thresholds
            acc_drop = original_metrics.accuracy - merged_metrics.accuracy
            rej_drop = original_metrics.rejection_rate - merged_metrics.rejection_rate
            fp_increase = merged_metrics.fp_rate - original_metrics.fp_rate
            
            print(f"      Original: Acc={original_metrics.accuracy:.1f}%, "
                  f"Rej={original_metrics.rejection_rate:.1f}%, FP={original_metrics.fp_rate:.1f}%")
            print(f"      Merged:   Acc={merged_metrics.accuracy:.1f}%, "
                  f"Rej={merged_metrics.rejection_rate:.1f}%, FP={merged_metrics.fp_rate:.1f}%")
            
            if (acc_drop > self.config.max_accuracy_drop or
                rej_drop > self.config.max_rejection_drop or
                fp_increase > self.config.max_fp_increase):
                
                print(f"      ❌ Failed (Acc↓{acc_drop:.1f}%, Rej↓{rej_drop:.1f}%, FP↑{fp_increase:.1f}%)")
                all_valid = False
                break
            else:
                print(f"      ✅ Passed")
        
        return all_valid
    
    def _get_task_data_from_buffer(self, memory_buffer, task_id: str, num_samples: int):
        """
        Get in-domain data for a task from memory buffer
        """
        all_data = memory_buffer.get_all_data()
        
        # Filter for this task
        task_indices = [i for i, tid in enumerate(all_data['task_ids']) 
                       if tid == task_id]
        
        if not task_indices:
            return None
        
        # Sample subset
        if len(task_indices) > num_samples:
            task_indices = random.sample(task_indices, num_samples)
        
        images = torch.stack([all_data['images'][i] for i in task_indices])
        labels = torch.tensor([all_data['labels'][i] for i in task_indices])
        
        return {'images': images, 'labels': labels}
    
    def group_adapters(self, all_loras: Dict[str, nn.Module]) -> List[List[str]]:
        """
        Group compatible LoRAs based on cosine similarity of their weight updates
        """
        # Calculate task vectors (flattened delta_W)
        lora_vectors = {}
        for task_id, lora in all_loras.items():
            delta_W = self._compute_delta_W(lora)
            lora_vectors[task_id] = delta_W.flatten()
        
        # Calculate pairwise cosine similarity
        similarity_matrix = {}
        task_ids = list(lora_vectors.keys())
        
        for i, task_i in enumerate(task_ids):
            similarity_matrix[task_i] = {}
            for j, task_j in enumerate(task_ids):
                if i == j:
                    similarity_matrix[task_i][task_j] = 1.0
                else:
                    vec_i = lora_vectors[task_i]
                    vec_j = lora_vectors[task_j]
                    cos_sim = F.cosine_similarity(vec_i.unsqueeze(0), 
                                                vec_j.unsqueeze(0)).item()
                    similarity_matrix[task_i][task_j] = cos_sim
        
        # Form groups based on threshold
        groups = []
        used_loras = set()
        
        for task_i in task_ids:
            if task_i not in used_loras:
                current_group = [task_i]
                used_loras.add(task_i)
                
                for task_j in task_ids:
                    if task_j not in used_loras:
                        if similarity_matrix[task_i][task_j] > self.config.similarity_threshold:
                            current_group.append(task_j)
                            used_loras.add(task_j)
                
                groups.append(current_group)
        
        print(f"Formed {len(groups)} groups from {len(all_loras)} LoRAs")
        for i, group in enumerate(groups):
            print(f"  Group {i}: {group}")
        
        return groups
    
    def _compute_delta_W(self, lora: nn.Module) -> torch.Tensor:
        """Extract the effective weight update from LoRA"""
        if hasattr(lora, 'lora_B') and hasattr(lora, 'lora_A'):
            # It's a LoRALayer
            return lora.lora_B @ lora.lora_A
        elif hasattr(lora, 'attention_adapters'):  
            # It's a TaskSpecificLoRA - get first layer's Q adapter as representative
            for adapter in lora.attention_adapters:
                if adapter is not None and 'q' in adapter.lora_modules:
                    q_lora = adapter.lora_modules['q']
                    return q_lora.lora_B @ q_lora.lora_A
        elif hasattr(lora, 'adapters'):
            # Alternative structure - get first adapter
            if len(lora.adapters) > 0 and 'q' in lora.adapters[0]:
                return lora.adapters[0]['q'].lora_B @ lora.adapters[0]['q'].lora_A
        
        raise ValueError(f"Unknown LoRA format: {type(lora)}")
    
    def _get_ood_data_from_buffer(self, memory_buffer, task_id: str, num_samples: int):
        """
        Get OOD data (from other tasks) from memory buffer
        """
        all_data = memory_buffer.get_all_data()
        
        # Filter for other tasks
        ood_indices = [i for i, tid in enumerate(all_data['task_ids']) 
                      if tid != task_id]
        
        if not ood_indices:
            return None
        
        # Sample subset
        if len(ood_indices) > num_samples:
            ood_indices = random.sample(ood_indices, num_samples)
        
        images = torch.stack([all_data['images'][i] for i in ood_indices])
        labels = torch.tensor([all_data['labels'][i] for i in ood_indices])
        
        return {'images': images, 'labels': labels}
    
    def _calculate_metrics_from_buffer(self,
                                      lora: nn.Module,
                                      in_domain_data: Dict,
                                      ood_data: Optional[Dict],
                                      backbone: nn.Module,
                                      task_head: nn.Module) -> PerformanceMetrics:
        """
        Calculate metrics using buffer data
        """
        device = next(backbone.parameters()).device
        
        # Temporarily inject LoRA
        original_state = self._save_backbone_state(backbone)
        self._inject_lora_into_backbone(lora, backbone)
        
        backbone.eval()
        task_head.eval()
        
        # In-domain accuracy
        correct_known = 0
        total_known = 0
        false_positives = 0
        
        with torch.no_grad():
            # Process in batches
            batch_size = 32
            images = in_domain_data['images']
            labels = in_domain_data['labels']
            
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i+batch_size].to(device)
                batch_labels = labels[i:i+batch_size].to(device)
                
                features = backbone(batch_images)
                logits = task_head(features)
                
                known_logits = logits[:, :-1]
                unknown_logits = logits[:, -1]
                
                predictions = torch.argmax(known_logits, dim=1)
                
                for j in range(len(batch_labels)):
                    if unknown_logits[j] < torch.max(known_logits[j]):
                        if predictions[j] == batch_labels[j]:
                            correct_known += 1
                        total_known += 1
                    else:
                        false_positives += 1
                        total_known += 1
        
        # OOD rejection rate
        correct_unknown = 0
        total_unknown = 0
        
        if ood_data:
            with torch.no_grad():
                images = ood_data['images']
                
                for i in range(0, len(images), batch_size):
                    batch_images = images[i:i+batch_size].to(device)
                    
                    features = backbone(batch_images)
                    logits = task_head(features)
                    
                    unknown_logits = logits[:, -1]
                    known_logits = logits[:, :-1]
                    
                    for j in range(len(batch_images)):
                        if unknown_logits[j] > torch.max(known_logits[j]):
                            correct_unknown += 1
                        total_unknown += 1
        
        # Restore backbone
        self._restore_backbone_state(backbone, original_state)
        
        # Calculate metrics
        accuracy = (correct_known / max(1, total_known)) * 100
        rejection_rate = (correct_unknown / max(1, total_unknown)) * 100 if total_unknown > 0 else 100.0
        fp_rate = (false_positives / max(1, total_known)) * 100
        
        return self.PerformanceMetrics(accuracy, rejection_rate, fp_rate)
    
    def _save_backbone_state(self, backbone):
        """Save backbone state before modification"""
        saved = {}
        for idx, block in enumerate(backbone.blocks):
            saved[idx] = {'attn': block.attn, 'mlp': block.mlp}
        return saved
    
    def _inject_lora_into_backbone(self, lora, backbone):
        """Inject LoRA for validation"""
        from src.models.lora_vit import LoRAAttention
        
        for idx, block in enumerate(backbone.blocks):
            if hasattr(block.attn, 'qkv'):
                hidden_dim = block.attn.qkv.in_features
                
                lora_attn = LoRAAttention(
                    original_attn=block.attn,
                    hidden_dim=hidden_dim,
                    lora_rank=lora.rank if hasattr(lora, 'rank') else 4,
                    lora_alpha=lora.alpha if hasattr(lora, 'alpha') else 4.0,
                    lora_dropout=0.1,
                    target_modules=["q", "v"]
                )
                
                lora_attn.lora_adapters["q"] = lora
                lora_attn.lora_adapters["v"] = lora
                
                block.attn = lora_attn
    
    def _restore_backbone_state(self, backbone, original_state):
        """Restore backbone to original state"""
        for idx, block in enumerate(backbone.blocks):
            if idx in original_state:
                block.attn = original_state[idx]['attn']
                block.mlp = original_state[idx]['mlp']

        

def integrate_intelligent_merge(hierarchical_model, validation_loaders=None):
    """
    Complete integration function
    """
    if len(hierarchical_model.active_block_tasks) < 2:
        print("Not enough tasks to merge")
        return [], []
    
    # Prepare validation data
    validation_data = {}
    if validation_loaders:
        for task_id in hierarchical_model.active_block_tasks:
            if task_id in validation_loaders:
                validation_data[task_id] = {
                    'in_domain': validation_loaders[task_id]['in_domain'],
                    'out_domain': validation_loaders[task_id]['ood']
                }
    
    # Store validation data in model
    for task_id, data in validation_data.items():
        hierarchical_model.task_validation_data[task_id] = data
    
    # Trigger merge
    success = hierarchical_model._intelligent_merge_current_block()
    
    if success:
        print("\n✅ Intelligent merge completed successfully!")
        return hierarchical_model.merged_blocks, []
    else:
        print("\n⚠️ Some merges failed validation, tasks kept separate")
        return hierarchical_model.merged_blocks, hierarchical_model.active_block_tasks.keys()
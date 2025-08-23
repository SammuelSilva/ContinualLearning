"""
Visualization tools for hierarchical LoRA structure and metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import torch
from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

class HierarchicalVisualizer:
    """
    Visualizes the hierarchical LoRA structure and training progress.
    """
    
    def __init__(self, save_dir: str = "./visualizations"):
        self.save_dir = save_dir
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        os.makedirs(self.save_dir, exist_ok=True) 

    def plot_hierarchy_tree(
        self,
        hierarchy_stats: Dict,
        save_path: Optional[str] = None
    ):
        """
        Create a tree visualization of the hierarchical structure.
        """
        fig = go.Figure()
        
        # Create tree structure
        G = nx.DiGraph()
        pos = {}
        labels = {}
        
        # Root node
        G.add_node("root", label="Model")
        pos["root"] = (0, 0)
        labels["root"] = "ViT Backbone"
        
        y_offset = -1.5
        x_spacing = 3
        
        # Add merged blocks
        for i, block in enumerate(hierarchy_stats['blocks']):
            block_node = f"block_{block['block_id']}"
            G.add_node(block_node, label=f"Block {block['block_id']}")
            G.add_edge("root", block_node)
            
            x_pos = (i - len(hierarchy_stats['blocks'])/2) * x_spacing
            pos[block_node] = (x_pos, y_offset)
            labels[block_node] = f"Block {block['block_id']}\n(Merged)"
            
            # Add tasks in block
            task_y = y_offset - 1.5
            task_x_spacing = x_spacing / (len(block['task_ids']) + 1)
            
            for j, task_id in enumerate(block['task_ids']):
                task_node = f"{block_node}_{task_id}"
                G.add_node(task_node, label=task_id)
                G.add_edge(block_node, task_node)
                
                task_x = x_pos + (j - len(block['task_ids'])/2) * task_x_spacing
                pos[task_node] = (task_x, task_y)
                labels[task_node] = task_id
        
        # Add active block if exists
        if 'active_block' in hierarchy_stats:
            active = hierarchy_stats['active_block']
            block_node = f"block_{active['block_id']}"
            G.add_node(block_node, label=f"Block {active['block_id']}")
            G.add_edge("root", block_node)
            
            x_pos = (len(hierarchy_stats['blocks']) - len(hierarchy_stats['blocks'])/2) * x_spacing
            pos[block_node] = (x_pos, y_offset)
            labels[block_node] = f"Block {active['block_id']}\n(Active)"
            
            # Add active tasks
            task_y = y_offset - 1.5
            task_x_spacing = x_spacing / (len(active['task_ids']) + 1)
            
            for j, task_id in enumerate(active['task_ids']):
                task_node = f"{block_node}_{task_id}"
                G.add_node(task_node, label=task_id)
                G.add_edge(block_node, task_node)
                
                task_x = x_pos + (j - len(active['task_ids'])/2) * task_x_spacing
                pos[task_node] = (task_x, task_y)
                labels[task_node] = task_id
        
        # Create edge trace
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=2, color='#888'),
                    hoverinfo='none'
                )
            )
        
        # Create node trace
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            text=[labels[node] for node in G.nodes()],
            textposition="top center",
            marker=dict(
                size=20,
                color=['red' if 'root' in node else 'blue' if 'block' in node and 'task' not in node else 'green' 
                       for node in G.nodes()],
                line=dict(width=2, color='white')
            ),
            hoverinfo='text'
        )
        
        # Create figure
        fig = go.Figure(data=edge_trace + [node_trace])
        fig.update_layout(
            title="Hierarchical LoRA Structure",
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def plot_memory_efficiency(
        self,
        num_tasks: int,
        tasks_per_block: int,
        lora_rank: int = 4,
        save_path: Optional[str] = None
    ):
        """
        Visualize memory efficiency of hierarchical vs standard approach.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        tasks = np.arange(1, num_tasks + 1)
        
        # Calculate memory usage
        standard_memory = tasks * lora_rank  # Linear growth
        hierarchical_memory = []
        
        for t in tasks:
            num_blocks = (t - 1) // tasks_per_block
            active_tasks = t - (num_blocks * tasks_per_block)
            # Merged blocks use higher rank but fewer modules
            block_memory = num_blocks * (lora_rank * 2)  # Assume 2x rank after merge
            active_memory = active_tasks * lora_rank
            hierarchical_memory.append(block_memory + active_memory)
        
        hierarchical_memory = np.array(hierarchical_memory)
        
        # Plot 1: Memory usage comparison
        ax = axes[0]
        ax.plot(tasks, standard_memory, 'b-', label='Standard', linewidth=2)
        ax.plot(tasks, hierarchical_memory, 'r--', label='Hierarchical', linewidth=2)
        ax.fill_between(tasks, 0, standard_memory, alpha=0.3, color='blue')
        ax.fill_between(tasks, 0, hierarchical_memory, alpha=0.3, color='red')
        ax.set_xlabel('Number of Tasks')
        ax.set_ylabel('Relative Memory Usage')
        ax.set_title('Memory Usage Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Memory savings
        ax = axes[1]
        savings = (standard_memory - hierarchical_memory) / standard_memory * 100
        ax.plot(tasks, savings, 'g-', linewidth=2)
        ax.fill_between(tasks, 0, savings, alpha=0.3, color='green')
        ax.set_xlabel('Number of Tasks')
        ax.set_ylabel('Memory Savings (%)')
        ax.set_title('Hierarchical Memory Savings')
        ax.grid(True, alpha=0.3)
        
        # Add annotations for key points
        merge_points = np.arange(tasks_per_block, num_tasks, tasks_per_block)
        for mp in merge_points:
            ax.axvline(x=mp, color='gray', linestyle=':', alpha=0.5)
            ax.text(mp, ax.get_ylim()[1] * 0.95, f'Merge {mp//tasks_per_block}',
                   ha='center', fontsize=9, color='gray')
        
        plt.suptitle('Hierarchical LoRA Memory Efficiency Analysis')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_orthogonality_matrix(
        self,
        adapters: Dict[str, torch.nn.Module],
        save_path: Optional[str] = None
    ):
        """
        Visualize orthogonality between different task adapters.
        """
        task_ids = list(adapters.keys())
        n_tasks = len(task_ids)
        
        # Compute orthogonality matrix
        ortho_matrix = np.zeros((n_tasks, n_tasks))
        
        for i, task_i in enumerate(task_ids):
            for j, task_j in enumerate(task_ids):
                if i == j:
                    ortho_matrix[i, j] = 1.0
                else:
                    # Compute orthogonality between adapters
                    adapter_i = adapters[task_i]
                    adapter_j = adapters[task_j]
                    
                    # Simple dot product based orthogonality
                    if hasattr(adapter_i, 'lora_A') and hasattr(adapter_j, 'lora_A'):
                        w_i = (adapter_i.lora_B @ adapter_i.lora_A).flatten()
                        w_j = (adapter_j.lora_B @ adapter_j.lora_A).flatten()
                        
                        w_i_norm = w_i / (torch.norm(w_i) + 1e-8)
                        w_j_norm = w_j / (torch.norm(w_j) + 1e-8)
                        
                        ortho_score = 1.0 - torch.abs(torch.dot(w_i_norm, w_j_norm)).item()
                        ortho_matrix[i, j] = ortho_score
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            ortho_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            xticklabels=task_ids,
            yticklabels=task_ids,
            cbar_kws={'label': 'Orthogonality Score'}
        )
        plt.title('Task Adapter Orthogonality Matrix')
        plt.xlabel('Task ID')
        plt.ylabel('Task ID')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def create_training_dashboard(
        self,
        metrics_history: Dict,
        hierarchy_stats: Dict,
        save_path: Optional[str] = None
    ):
        """
        Create a comprehensive training dashboard.
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Task Accuracy Evolution',
                'Memory Usage',
                'Orthogonality Score',
                'Forgetting Analysis',
                'Block Performance',
                'Hierarchical Structure'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'scatter'}],
                [{'type': 'heatmap'}, {'type': 'bar'}, {'type': 'scatter'}]
            ]
        )
        
        # Add traces for each subplot
        # ... (implement based on available metrics)
        
        fig.update_layout(height=800, showlegend=True)
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()


class MetricsAnimator:
    """
    Creates animated visualizations of training progress.
    """
    
    @staticmethod
    def create_accuracy_animation(
        accuracy_matrix: np.ndarray,
        save_path: str
    ):
        """
        Create an animated heatmap showing accuracy evolution.
        """
        import matplotlib.animation as animation
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def animate(frame):
            ax.clear()
            
            # Create partial matrix up to current frame
            partial_matrix = np.full_like(accuracy_matrix, np.nan)
            partial_matrix[:frame+1, :frame+1] = accuracy_matrix[:frame+1, :frame+1]
            
            # Plot heatmap
            sns.heatmap(
                partial_matrix,
                annot=True,
                fmt='.1f',
                cmap='YlOrRd',
                vmin=0,
                vmax=100,
                ax=ax,
                cbar_kws={'label': 'Accuracy (%)'}
            )
            
            ax.set_title(f'Task Accuracy Evolution - After Task {frame}')
            ax.set_xlabel('Task Evaluated')
            ax.set_ylabel('Training Progress')
        
        anim = animation.FuncAnimation(
            fig, animate, frames=accuracy_matrix.shape[0],
            interval=500, repeat=True
        )
        
        anim.save(save_path, writer='pillow')
        plt.close()
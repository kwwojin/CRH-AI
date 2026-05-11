import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Any, List, Optional


class CRHVisualizer:
    @staticmethod
    def plot_manifold(manifold, title: str = "CRH Manifold Visualization") -> plt.Figure:
        if manifold.coords is None or len(manifold.coords) < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Insufficient data to plot", ha='center', va='center')
            return fig
        
        coords = manifold.coords
        
        if coords.shape[1] >= 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='blue', s=100, alpha=0.6)
            
            if manifold.R_bg is not None:
                colors = plt.cm.viridis(manifold.R_bg / np.max(manifold.R_bg))
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors, s=100, alpha=0.8)
            
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_zlabel('Dimension 3')
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(coords[:, 0], coords[:, 1], c='blue', s=100, alpha=0.6)
            
            if manifold.R_bg is not None:
                colors = plt.cm.viridis(manifold.R_bg / np.max(manifold.R_bg))
                ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=100, alpha=0.8)
            
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
        
        plt.title(title)
        plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), label='Curvature')
        
        return fig
    
    @staticmethod
    def plot_entanglement_field(entanglement_field, query_point: np.ndarray = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if entanglement_field.manifold.coords is not None:
            coords = entanglement_field.manifold.coords
            ax.scatter(coords[:, 0], coords[:, 1], c='gray', s=50, alpha=0.5, label='Entities')
        
        if query_point is not None:
            ax.scatter(query_point[0], query_point[1], c='red', s=200, marker='*', label='Query')
            
            if entanglement_field.manifold.coords is not None:
                for coord in entanglement_field.manifold.coords:
                    strength = entanglement_field.compute_gradient(
                        {'coordinates': query_point},
                        {'coordinates': coord}
                    )
                    if strength > 0.1:
                        ax.arrow(query_point[0], query_point[1],
                                coord[0] - query_point[0],
                                coord[1] - query_point[1],
                                alpha=strength, color='green', width=0.01)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.title('Entanglement Field Visualization')
        plt.legend()
        
        return fig
    
    @staticmethod
    def plot_coherence_history(entropy_balance, title: str = "Self-Consistency History") -> plt.Figure:
        history = entropy_balance.self_consistency_history
        
        if not history:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No history data", ha='center', va='center')
            return fig
        
        scores = [h['score'] for h in history]
        timestamps = range(len(scores))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(timestamps, scores, 'b-', linewidth=2, label='Self-Consistency')
        ax.axhline(y=entropy_balance.target_coherence, color='r', linestyle='--', label=f'Target: {entropy_balance.target_coherence}')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Self-Consistency Score')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        return fig
    
    @staticmethod
    def plot_constraint_violations(constraints, title: str = "Constraint Violations") -> plt.Figure:
        causal_violations = len(constraints.causal_constraint.get_violations())
        topological_violations = len(constraints.topological_constraint.get_violations())
        
        fig, ax = plt.subplots(figsize=(8, 6))
        categories = ['Causal', 'Topological']
        counts = [causal_violations, topological_violations]
        
        ax.bar(categories, counts, color=['red', 'orange'])
        ax.set_xlabel('Constraint Type')
        ax.set_ylabel('Number of Violations')
        ax.set_title(title)
        ax.set_ylim(0, max(counts) + 1)
        
        return fig
    
    @staticmethod
    def plot_entropy_balance(entropy_balance, title: str = "Entropy Balance Evolution") -> plt.Figure:
        eta_history = entropy_balance.eta_history
        
        if not eta_history:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No entropy data", ha='center', va='center')
            return fig
        
        etas = [h['eta'] for h in eta_history]
        consistencies = [h.get('consistency', 0.0) for h in eta_history]
        timestamps = range(len(etas))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(timestamps, etas, 'g-', linewidth=2, label='Entropy Balance (η)')
        ax1.axhline(y=1.0, color='r', linestyle='--')
        ax1.set_ylabel('η')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(timestamps, consistencies, 'b-', linewidth=2, label='Self-Consistency')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Consistency Score')
        ax2.legend()
        ax2.grid(True)
        
        plt.suptitle(title)
        
        return fig
    
    @staticmethod
    def visualize_system_state(crh_base, filename: Optional[str] = None):
        fig1 = CRHVisualizer.plot_manifold(crh_base.manifold)
        fig2 = CRHVisualizer.plot_coherence_history(crh_base.entropy_balance)
        fig3 = CRHVisualizer.plot_entropy_balance(crh_base.entropy_balance)
        
        if filename:
            fig1.savefig(f"{filename}_manifold.png")
            fig2.savefig(f"{filename}_coherence.png")
            fig3.savefig(f"{filename}_entropy.png")
            plt.close('all')
        else:
            plt.show()
        
        return [fig1, fig2, fig3]

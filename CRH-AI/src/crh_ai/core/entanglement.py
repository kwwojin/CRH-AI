import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from typing import Dict, Any, List, Optional


class EntanglementField:
    def __init__(self, manifold, scale: float = 1.0):
        self.manifold = manifold
        self.scale = scale
        self.gradient_history = []
    
    def _compute_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        x_norm = x / np.sum(x) if np.sum(x) > 0 else x
        y_norm = y / np.sum(y) if np.sum(y) > 0 else y
        
        h_x = entropy(x_norm + 1e-10)
        h_y = entropy(y_norm + 1e-10)
        
        joint = np.concatenate([x, y])
        joint_norm = joint / np.sum(joint) if np.sum(joint) > 0 else joint
        h_xy = entropy(joint_norm + 1e-10)
        
        return h_x + h_y - h_xy
    
    def compute_gradient(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> float:
        if 'coordinates' not in entity1 or 'coordinates' not in entity2:
            return 0.0
        
        pos1 = entity1['coordinates']
        pos2 = entity2['coordinates']
        
        dist = np.linalg.norm(pos1 - pos2)
        
        coherence1 = entity1.get('crh_density', 1.0)
        coherence2 = entity2.get('crh_density', 1.0)
        
        avg_coherence = (coherence1 + coherence2) / 2
        
        strength = np.exp(-dist / self.scale) * avg_coherence
        
        return strength
    
    def perceive_field(self, query: Dict[str, Any], context: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        if 'coordinates' not in query:
            return {'gradient_tensor': [], 'strongest_connection': None, 'avg_entanglement': 0.0}
        
        if context is None or not context:
            return {'gradient_tensor': [], 'strongest_connection': None, 'avg_entanglement': 0.0}
        
        query_coord = query['coordinates']
        context_coords = np.array([ctx['coordinates'] for ctx in context])
        
        distances = cdist(query_coord.reshape(1, -1), context_coords)[0]
        
        gradient_tensor = []
        
        for i, ctx in enumerate(context):
            ctx_coord = ctx['coordinates']
            distance = distances[i]
            
            mi = self._compute_mutual_information(query_coord, ctx_coord)
            entanglement = self.compute_gradient(query, ctx)
            
            gradient_tensor.append({
                'index': i,
                'distance': distance,
                'entanglement_strength': entanglement,
                'mutual_information': mi,
                'direction': (ctx_coord - query_coord) / (distance + 1e-10),
                'weight': entanglement / (distance + 1e-10)
            })
        
        gradient_tensor.sort(key=lambda x: -x['entanglement_strength'])
        
        self.gradient_history.append({
            'timestamp': np.datetime64('now'),
            'gradient_tensor': gradient_tensor
        })
        
        avg_entanglement = np.mean([g['entanglement_strength'] for g in gradient_tensor]) if gradient_tensor else 0.0
        
        return {
            'gradient_tensor': gradient_tensor,
            'strongest_connection': gradient_tensor[0] if gradient_tensor else None,
            'avg_entanglement': avg_entanglement,
            'field_strength': np.sum([g['entanglement_strength'] for g in gradient_tensor])
        }
    
    def update_scale(self, new_scale: float):
        self.scale = new_scale
    
    def get_field_state(self) -> Dict[str, Any]:
        if not self.gradient_history:
            return {'history_length': 0, 'current_strength': 0.0}
        
        recent = self.gradient_history[-1]
        strengths = [g['entanglement_strength'] for g in recent['gradient_tensor']]
        
        return {
            'history_length': len(self.gradient_history),
            'current_strength': np.mean(strengths) if strengths else 0.0,
            'peak_strength': np.max(strengths) if strengths else 0.0,
            'field_balance': np.var(strengths) if len(strengths) > 1 else 0.0
        }

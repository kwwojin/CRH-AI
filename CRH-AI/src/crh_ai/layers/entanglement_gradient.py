import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from typing import Dict, Any, List, Optional


class EntanglementGradientLayer:
    def __init__(self, E0: float = 1.0, beta: float = 0.5):
        self.E0 = E0
        self.beta = beta
        
        self.entanglement_tensor = {}
        self.coherence_scores = {}
        self.field_distribution = None
    
    def _compute_entropy(self, data: np.ndarray) -> float:
        if len(data) == 0:
            return 0.0
        normalized = data / np.sum(data) if np.sum(data) > 0 else data
        return entropy(normalized + 1e-10)
    
    def _compute_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        h_x = self._compute_entropy(x)
        h_y = self._compute_entropy(y)
        
        joint = np.concatenate([x, y])
        h_xy = self._compute_entropy(joint)
        
        return h_x + h_y - h_xy
    
    def _compute_entanglement_strength(self, distance: float, mi: float = 0.0) -> float:
        distance_factor = self.E0 * np.exp(-self.beta * distance)
        return distance_factor * (1 + mi)
    
    def perceive_gradient(self, query_geometry: Dict[str, Any], context_geometries: List[Dict[str, Any]]) -> Dict[str, Any]:
        if 'coordinates' not in query_geometry:
            raise ValueError("Query geometry must contain 'coordinates'")
        
        if not context_geometries:
            return {
                'gradient_tensor': [],
                'strongest_connection': None,
                'avg_entanglement': 0.0
            }
        
        query_coord = query_geometry['coordinates']
        query_curvature = query_geometry.get('curvature', 1.0)
        
        context_coords = np.array([ctx['coordinates'] for ctx in context_geometries])
        distances = cdist(query_coord.reshape(1, -1), context_coords)[0]
        
        gradient_tensor = []
        
        for i, ctx in enumerate(context_geometries):
            ctx_coord = ctx['coordinates']
            ctx_curvature = ctx.get('curvature', 1.0)
            
            mi = self._compute_mutual_information(query_coord, ctx_coord)
            distance = distances[i]
            entanglement = self._compute_entanglement_strength(distance, mi)
            
            gradient_tensor.append({
                'index': i,
                'distance': distance,
                'entanglement_strength': entanglement,
                'mutual_information': mi,
                'curvature_ratio': ctx_curvature / query_curvature
            })
        
        gradient_tensor.sort(key=lambda x: -x['entanglement_strength'])
        
        self._update_field_distribution(gradient_tensor)
        
        return {
            'gradient_tensor': gradient_tensor,
            'strongest_connection': gradient_tensor[0] if gradient_tensor else None,
            'avg_entanglement': np.mean([g['entanglement_strength'] for g in gradient_tensor]) if gradient_tensor else 0.0
        }
    
    def compute_coherence_score(self, geometric_reprs: List[Dict[str, Any]]) -> Dict[str, float]:
        if len(geometric_reprs) < 2:
            return {'global_coherence': 1.0, 'local_coherence': 1.0, 'avg_distance': 0.0, 'curvature_std': 0.0}
        
        coords = np.array([g['coordinates'] for g in geometric_reprs])
        curvatures = np.array([g.get('curvature', 1.0) for g in geometric_reprs])
        redshifts = np.array([np.exp(c) - 1 for c in curvatures])
        
        all_distances = cdist(coords, coords)
        np.fill_diagonal(all_distances, np.inf)
        avg_distance = np.mean(all_distances)
        
        ci_values = []
        for i in range(len(geometric_reprs)):
            r_bg_i = curvatures[i]
            z_i = redshifts[i]
            ci = 1 - 2 * np.abs((r_bg_i - z_i) / (1 + z_i + 1e-10))
            ci_values.append(ci)
        
        global_coherence = np.mean(ci_values)
        
        local_coherence_sum = 0.0
        for i in range(len(geometric_reprs)):
            neighbors = [j for j in range(len(geometric_reprs)) if all_distances[i, j] < avg_distance]
            if neighbors:
                local_ci = np.mean([ci_values[j] for j in neighbors])
                local_coherence_sum += local_ci
        local_coherence = local_coherence_sum / len(geometric_reprs)
        
        self.coherence_scores['global'] = global_coherence
        self.coherence_scores['local'] = local_coherence
        
        return {
            'global_coherence': global_coherence,
            'local_coherence': local_coherence,
            'avg_distance': avg_distance,
            'curvature_std': np.std(curvatures)
        }
    
    def _update_field_distribution(self, gradient_tensor: List[Dict[str, Any]]):
        if not gradient_tensor:
            self.field_distribution = None
            return
        
        strengths = np.array([g['entanglement_strength'] for g in gradient_tensor])
        distances = np.array([g['distance'] for g in gradient_tensor])
        
        self.field_distribution = {
            'high_coherence_region': np.mean(strengths[strengths > np.mean(strengths)]),
            'low_coherence_region': np.mean(strengths[strengths <= np.mean(strengths)]),
            'peak_strength': np.max(strengths),
            'field_balance': np.mean(strengths) / np.mean(distances) if np.mean(distances) > 0 else 0
        }
    
    def get_field_state(self) -> Optional[Dict[str, Any]]:
        return self.field_distribution
    
    def get_coherence_history(self) -> Dict[str, float]:
        return self.coherence_scores

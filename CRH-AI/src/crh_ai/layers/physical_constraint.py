import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict, Any, List, Tuple, Optional


class PhysicalConstraintLayer:
    def __init__(self, coherence_threshold: float = 0.8, collision_threshold: float = 0.1):
        self.coherence_threshold = coherence_threshold
        self.collision_threshold = collision_threshold
        
        self.violation_history = []
        self.constraint_weights = {
            'temporal': 1.0,
            'spatial': 1.0,
            'conservation': 1.0,
            'topological': 1.0
        }
    
    def _check_temporal_constraint(self, geometric_reprs: List[Dict[str, Any]]) -> Tuple[bool, str, float]:
        if len(geometric_reprs) < 2:
            return True, "Not enough entities for temporal check", 0.0
        
        phases = np.array([g.get('phase', 0.0) for g in geometric_reprs])
        
        for i in range(len(phases)):
            for j in range(i + 1, len(phases)):
                phase_diff = phases[j] - phases[i]
                if phase_diff < -np.pi:
                    phase_diff += 2 * np.pi
                if phase_diff < 0:
                    penalty = abs(phase_diff) / np.pi
                    return False, f"Temporal violation: entity {i} has higher phase than entity {j}", penalty
        
        return True, "No temporal violations", 0.0
    
    def _check_spatial_constraint(self, geometric_reprs: List[Dict[str, Any]]) -> Tuple[bool, str, float]:
        if len(geometric_reprs) < 2:
            return True, "Not enough entities for spatial check", 0.0
        
        coords = np.array([g['coordinates'] for g in geometric_reprs])
        distances = cdist(coords, coords)
        np.fill_diagonal(distances, np.inf)
        
        min_distance = np.min(distances)
        
        if min_distance < self.collision_threshold:
            penalty = (self.collision_threshold - min_distance) / self.collision_threshold
            indices = np.where(distances == min_distance)
            i, j = indices[0][0], indices[1][0]
            return False, f"Spatial violation: entities {i} and {j} are overlapping", penalty
        
        return True, "No spatial violations", 0.0
    
    def _check_conservation_constraint(self, geometric_reprs: List[Dict[str, Any]]) -> Tuple[bool, str, float]:
        if len(geometric_reprs) < 1:
            return True, "No entities to check", 0.0
        
        total_curvature = np.sum([g.get('curvature', 0.0) for g in geometric_reprs])
        total_topological_charge = np.sum([g.get('topological_charge', 0.0) for g in geometric_reprs])
        
        expected_curvature = len(geometric_reprs) * 0.5
        expected_charge = len(geometric_reprs) * 0.8
        
        curvature_deviation = abs(total_curvature - expected_curvature) / expected_curvature
        charge_deviation = abs(total_topological_charge - expected_charge) / expected_charge
        
        max_deviation = max(curvature_deviation, charge_deviation)
        
        if max_deviation > 0.5:
            penalty = max_deviation
            return False, f"Conservation violation: curvature={total_curvature:.2f}, charge={total_topological_charge:.2f}", penalty
        
        return True, "No conservation violations", 0.0
    
    def _check_topological_constraint(self, geometric_reprs: List[Dict[str, Any]]) -> Tuple[bool, str, float]:
        if len(geometric_reprs) < 1:
            return True, "No entities to check", 0.0
        
        charges = np.array([g.get('topological_charge', 0.0) for g in geometric_reprs])
        high_charge_mask = charges > 0.8
        
        if np.sum(high_charge_mask) > 0:
            high_charges = charges[high_charge_mask]
            charge_std = np.std(high_charges)
            
            if charge_std > 0.2:
                penalty = charge_std / 0.2
                return False, "Topological violation: high charge entities show instability", penalty
        
        return True, "No topological violations", 0.0
    
    def _check_coherence_threshold(self, global_coherence: float) -> Tuple[bool, str, float]:
        if global_coherence < self.coherence_threshold:
            penalty = (self.coherence_threshold - global_coherence) / self.coherence_threshold
            return False, f"Coherence violation: {global_coherence:.4f} < {self.coherence_threshold}", penalty
        return True, "Coherence threshold met", 0.0
    
    def check_constraint(self, geometric_reprs: List[Dict[str, Any]], global_coherence: Optional[float] = None) -> Dict[str, Any]:
        violations = []
        total_penalty = 0.0
        
        temporal_ok, temporal_msg, temporal_penalty = self._check_temporal_constraint(geometric_reprs)
        if not temporal_ok:
            violations.append({'type': 'temporal', 'message': temporal_msg})
            total_penalty += temporal_penalty * self.constraint_weights['temporal']
        
        spatial_ok, spatial_msg, spatial_penalty = self._check_spatial_constraint(geometric_reprs)
        if not spatial_ok:
            violations.append({'type': 'spatial', 'message': spatial_msg})
            total_penalty += spatial_penalty * self.constraint_weights['spatial']
        
        conservation_ok, conservation_msg, conservation_penalty = self._check_conservation_constraint(geometric_reprs)
        if not conservation_ok:
            violations.append({'type': 'conservation', 'message': conservation_msg})
            total_penalty += conservation_penalty * self.constraint_weights['conservation']
        
        topological_ok, topological_msg, topological_penalty = self._check_topological_constraint(geometric_reprs)
        if not topological_ok:
            violations.append({'type': 'topological', 'message': topological_msg})
            total_penalty += topological_penalty * self.constraint_weights['topological']
        
        if global_coherence is not None:
            coherence_ok, coherence_msg, coherence_penalty = self._check_coherence_threshold(global_coherence)
            if not coherence_ok:
                violations.append({'type': 'coherence', 'message': coherence_msg})
                total_penalty += coherence_penalty
        
        self.violation_history.append({
            'timestamp': np.datetime64('now'),
            'violations': violations,
            'total_penalty': total_penalty
        })
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'penalty_score': total_penalty,
            'num_violations': len(violations)
        }
    
    def get_violation_history(self) -> List[Dict[str, Any]]:
        return self.violation_history
    
    def set_constraint_weights(self, weights: Dict[str, float]):
        self.constraint_weights.update(weights)
    
    def set_coherence_threshold(self, threshold: float):
        self.coherence_threshold = threshold

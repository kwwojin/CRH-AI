import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict, Any, List, Tuple


class TopologicalConstraint:
    def __init__(self, collision_threshold: float = 0.1):
        self.collision_threshold = collision_threshold
        self.violations = []
    
    def check_overlap(self, entities: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        violations = []
        
        if len(entities) < 2:
            return True, violations
        
        coords = np.array([e['coordinates'] for e in entities if 'coordinates' in e])
        
        if len(coords) < 2:
            return True, violations
        
        distances = cdist(coords, coords)
        np.fill_diagonal(distances, np.inf)
        
        min_distance = np.min(distances)
        
        if min_distance < self.collision_threshold:
            indices = np.where(distances == min_distance)
            i, j = indices[0][0], indices[1][0]
            message = f"Topological overlap: entities {i} and {j} are at distance {min_distance:.4f} < threshold {self.collision_threshold}"
            violations.append(message)
        
        self.violations.extend(violations)
        
        return len(violations) == 0, violations
    
    def check_connectivity(self, entities: List[Dict[str, Any]], max_gap: float = 1.0) -> Tuple[bool, List[str]]:
        violations = []
        
        if len(entities) < 2:
            return True, violations
        
        coords = np.array([e['coordinates'] for e in entities if 'coordinates' in e])
        
        if len(coords) < 2:
            return True, violations
        
        distances = cdist(coords, coords)
        np.fill_diagonal(distances, np.inf)
        
        avg_distance = np.mean(distances)
        
        if avg_distance > max_gap:
            message = f"Disconnected topology: average distance {avg_distance:.4f} > max gap {max_gap}"
            violations.append(message)
        
        self.violations.extend(violations)
        
        return len(violations) == 0, violations
    
    def compute_topological_charge(self, entity: Dict[str, Any]) -> float:
        if 'coordinates' not in entity:
            return 0.0
        
        coords = entity['coordinates']
        normalized = coords / np.linalg.norm(coords) if np.linalg.norm(coords) > 0 else coords
        
        return np.sum(normalized ** 2)
    
    def check_topological_stability(self, entities: List[Dict[str, Any]], min_charge: float = 0.8) -> Tuple[bool, List[str]]:
        violations = []
        
        for i, entity in enumerate(entities):
            charge = self.compute_topological_charge(entity)
            
            if charge < min_charge:
                message = f"Topological instability: entity {i} has charge {charge:.4f} < threshold {min_charge}"
                violations.append(message)
        
        self.violations.extend(violations)
        
        return len(violations) == 0, violations
    
    def validate_topology(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        overlap_valid, overlap_violations = self.check_overlap(entities)
        connectivity_valid, connectivity_violations = self.check_connectivity(entities)
        stability_valid, stability_violations = self.check_topological_stability(entities)
        
        all_violations = overlap_violations + connectivity_violations + stability_violations
        
        return {
            'valid': overlap_valid and connectivity_valid and stability_valid,
            'overlap_check': {'valid': overlap_valid, 'violations': overlap_violations},
            'connectivity_check': {'valid': connectivity_valid, 'violations': connectivity_violations},
            'stability_check': {'valid': stability_valid, 'violations': stability_violations},
            'total_violations': len(all_violations)
        }
    
    def get_violations(self) -> List[str]:
        return self.violations
    
    def clear_violations(self):
        self.violations = []

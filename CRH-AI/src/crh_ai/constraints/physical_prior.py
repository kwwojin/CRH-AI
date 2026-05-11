import numpy as np
from typing import Dict, Any, List, Tuple
from .causal import CausalConstraint
from .topological import TopologicalConstraint


class PhysicalConstraints:
    def __init__(self, coherence_threshold: float = 0.8, collision_threshold: float = 0.1):
        self.causal_constraint = CausalConstraint()
        self.topological_constraint = TopologicalConstraint(collision_threshold)
        self.coherence_threshold = coherence_threshold
        self.self_consistency_score = 1.0
        self.penalty_factor = 0.1
    
    def check_causal(self, events: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        return self.causal_constraint.check_temporal_order(events)
    
    def check_topological(self, entities: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        overlap_valid, overlap_violations = self.topological_constraint.check_overlap(entities)
        stability_valid, stability_violations = self.topological_constraint.check_topological_stability(entities)
        
        all_violations = overlap_violations + stability_violations
        
        return overlap_valid and stability_valid, all_violations
    
    def check_conservation(self, state_before: Dict[str, Any], state_after: Dict[str, Any]) -> Tuple[bool, List[str]]:
        violations = []
        
        energy_before = state_before.get('energy', 1.0)
        energy_after = state_after.get('energy', 1.0)
        
        if abs(energy_before - energy_after) > 0.1 * energy_before:
            violations.append(f"Energy conservation violation: {energy_before:.4f} → {energy_after:.4f}")
        
        info_before = state_before.get('information', 1.0)
        info_after = state_after.get('information', 1.0)
        
        if info_after < 0.5 * info_before:
            violations.append(f"Information loss: {info_before:.4f} → {info_after:.4f}")
        
        return len(violations) == 0, violations
    
    def check_self_consistency(self, consistency_score: float) -> Tuple[bool, str]:
        if consistency_score < self.coherence_threshold:
            message = f"Self-consistency violation: {consistency_score:.4f} < threshold {self.coherence_threshold}"
            return False, message
        
        return True, "Self-consistency OK"
    
    def check_consistency(self, proposed_action: Dict[str, Any], current_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        violations = []
        
        if 'events' in proposed_action:
            causal_valid, causal_violations = self.check_causal(proposed_action['events'])
            if not causal_valid:
                violations.extend(causal_violations)
        
        if 'entities' in proposed_action:
            topological_valid, topological_violations = self.check_topological(proposed_action['entities'])
            if not topological_valid:
                violations.extend(topological_violations)
        
        if 'state_after' in proposed_action:
            conservation_valid, conservation_violations = self.check_conservation(current_state, proposed_action['state_after'])
            if not conservation_valid:
                violations.extend(conservation_violations)
        
        if 'consistency_score' in proposed_action:
            consistency_valid, consistency_msg = self.check_self_consistency(proposed_action['consistency_score'])
            if not consistency_valid:
                violations.append(consistency_msg)
        
        if violations:
            penalty = len(violations) * self.penalty_factor
            self.self_consistency_score = max(0.0, self.self_consistency_score - penalty)
            return False, violations
        
        self.self_consistency_score = min(1.0, self.self_consistency_score + 0.01)
        
        return True, violations
    
    def validate_all(self, geometric_reprs: List[Dict[str, Any]], consistency_score: float = 1.0) -> Dict[str, Any]:
        proposed_action = {
            'entities': geometric_reprs,
            'consistency_score': consistency_score
        }
        current_state = {'energy': 1.0, 'information': len(geometric_reprs)}
        
        valid, violations = self.check_consistency(proposed_action, current_state)
        
        return {
            'valid': valid,
            'violations': violations,
            'self_consistency_score': self.self_consistency_score,
            'num_violations': len(violations)
        }
    
    def get_self_consistency_score(self) -> float:
        return self.self_consistency_score
    
    def reset_score(self):
        self.self_consistency_score = 1.0
        self.causal_constraint.clear_violations()
        self.topological_constraint.clear_violations()

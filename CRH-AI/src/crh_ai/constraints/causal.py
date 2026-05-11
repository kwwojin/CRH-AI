import numpy as np
from typing import Dict, Any, List, Tuple


class CausalConstraint:
    def __init__(self, c: float = 1.0):
        self.c = c  
        self.violations = []
    
    def check_temporal_order(self, events: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        violations = []
        
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                event_i = events[i]
                event_j = events[j]
                
                if 't' not in event_i or 't' not in event_j:
                    continue
                
                t_i = event_i['t']
                t_j = event_j['t']
                
                if 'x' in event_i and 'x' in event_j:
                    dx = np.linalg.norm(event_j['x'] - event_i['x'])
                    dt = t_j - t_i
                    
                    if dt < 0:
                        violations.append(f"Causal violation: event {j} before event {i} (Δt = {dt:.4f})")
                    elif dt < dx / self.c:
                        violations.append(f"Speed of light violation: event {j} too far from event {i}")
        
        self.violations.extend(violations)
        
        return len(violations) == 0, violations
    
    def check_causality(self, cause: Dict[str, Any], effect: Dict[str, Any]) -> Tuple[bool, str]:
        if 't' not in cause or 't' not in effect:
            return True, "No timestamp information"
        
        if cause['t'] > effect['t']:
            message = f"Causal reversal: cause time ({cause['t']}) > effect time ({effect['t']})"
            self.violations.append(message)
            return False, message
        
        if 'x' in cause and 'x' in effect:
            dx = np.linalg.norm(effect['x'] - cause['x'])
            dt = effect['t'] - cause['t']
            
            if dt < dx / self.c:
                message = f"Non-local effect: distance {dx:.4f} > c × Δt {self.c * dt:.4f}"
                self.violations.append(message)
                return False, message
        
        return True, "Causally consistent"
    
    def validate_sequence(self, sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = []
        
        for i in range(len(sequence) - 1):
            valid, message = self.check_causality(sequence[i], sequence[i + 1])
            results.append({
                'pair': (i, i + 1),
                'valid': valid,
                'message': message
            })
        
        all_valid = all(r['valid'] for r in results)
        
        return {
            'valid': all_valid,
            'checks': results,
            'num_violations': len([r for r in results if not r['valid']])
        }
    
    def get_violations(self) -> List[str]:
        return self.violations
    
    def clear_violations(self):
        self.violations = []

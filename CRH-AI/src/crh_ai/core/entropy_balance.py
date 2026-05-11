import numpy as np
from scipy.stats import entropy
from typing import Dict, Any, List


class EntropyBalance:
    def __init__(self, target_coherence: float = 0.85):
        self.target_coherence = target_coherence
        self.eta_history = []
        self.self_consistency_history = []
    
    def _compute_shannon_entropy(self, data: np.ndarray) -> float:
        if len(data) == 0:
            return 0.0
        normalized = data / np.sum(data) if np.sum(data) > 0 else data
        return entropy(normalized + 1e-10)
    
    def _compute_mdl_score(self, geometric_reprs: List[Dict[str, Any]]) -> float:
        if len(geometric_reprs) < 2:
            return 1.0
        
        coords = np.array([g['coordinates'] for g in geometric_reprs])
        curvatures = np.array([g.get('curvature', 1.0) for g in geometric_reprs])
        
        coord_entropy = self._compute_shannon_entropy(coords.flatten())
        curvature_entropy = self._compute_shannon_entropy(curvatures)
        
        avg_entropy = (coord_entropy + curvature_entropy) / 2
        
        return np.exp(-avg_entropy)
    
    def evaluate_self_consistency(self, geometric_reprs: List[Dict[str, Any]]) -> float:
        if len(geometric_reprs) < 1:
            return 0.0
        
        mdl_score = self._compute_mdl_score(geometric_reprs)
        
        curvatures = np.array([g.get('curvature', 1.0) for g in geometric_reprs])
        redshifts = np.array([np.exp(c) - 1 for c in curvatures])
        
        ci_values = []
        for i in range(len(geometric_reprs)):
            r_bg_i = curvatures[i]
            z_i = redshifts[i]
            ci = 1 - 2 * np.abs((r_bg_i - z_i) / (1 + z_i + 1e-10))
            ci_values.append(ci)
        
        crh_consistency = np.mean(ci_values)
        
        final_score = (mdl_score + crh_consistency) / 2
        
        self.self_consistency_history.append({
            'timestamp': np.datetime64('now'),
            'score': final_score,
            'mdl_score': mdl_score,
            'crh_consistency': crh_consistency
        })
        
        return final_score
    
    def compute_entropy_balance(self, geometric_reprs: List[Dict[str, Any]]) -> float:
        consistency = self.evaluate_self_consistency(geometric_reprs)
        
        eta = consistency / self.target_coherence
        
        if eta > 1.5:
            eta = 1.5
        elif eta < 0.5:
            eta = 0.5
        
        self.eta_history.append({
            'timestamp': np.datetime64('now'),
            'eta': eta,
            'consistency': consistency
        })
        
        return eta
    
    def get_entropy_state(self) -> Dict[str, Any]:
        if not self.eta_history:
            return {
                'current_eta': 1.0,
                'avg_eta': 1.0,
                'consistency_trend': 'stable',
                'balance_status': 'normal'
            }
        
        recent_etas = [h['eta'] for h in self.eta_history[-10:]]
        recent_consistencies = [h['consistency'] for h in self.self_consistency_history[-10:]]
        
        avg_eta = np.mean(recent_etas)
        avg_consistency = np.mean(recent_consistencies) if recent_consistencies else 0.0
        
        if len(recent_etas) >= 3:
            trend = np.polyfit(range(len(recent_etas)), recent_etas, 1)[0]
            if trend > 0.01:
                consistency_trend = 'increasing'
            elif trend < -0.01:
                consistency_trend = 'decreasing'
            else:
                consistency_trend = 'stable'
        else:
            consistency_trend = 'stable'
        
        balance_status = 'normal'
        if avg_consistency < 0.6:
            balance_status = 'warning'
        elif avg_consistency < 0.4:
            balance_status = 'critical'
        
        return {
            'current_eta': self.eta_history[-1]['eta'],
            'avg_eta': avg_eta,
            'consistency_trend': consistency_trend,
            'balance_status': balance_status,
            'avg_consistency': avg_consistency
        }
    
    def reset(self):
        self.eta_history = []
        self.self_consistency_history = []

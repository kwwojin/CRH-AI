import numpy as np
from typing import Dict, Any, List, Optional, Callable
from .core.manifold import CRHManifold
from .core.geometry import GeometricEngine
from .core.entanglement import EntanglementField
from .core.entropy_balance import EntropyBalance
from .constraints.physical_prior import PhysicalConstraints
from .reconstruction.mapper import WeakCoherenceMapper


class CRHGeometricBase:
    def __init__(self, dim: int = 64, coherence_threshold: float = 0.8, learning_rate: float = 0.015):
        self.manifold = CRHManifold(dim=dim)
        self.geometry = GeometricEngine()
        self.entanglement_field = EntanglementField(self.manifold)
        self.entropy_balance = EntropyBalance(target_coherence=coherence_threshold)
        self.constraints = PhysicalConstraints(coherence_threshold=coherence_threshold)
        self.mapper = WeakCoherenceMapper(self.manifold, self.entanglement_field)
        
        self.learning_rate = learning_rate
        self.history = []
        self.current_context = {}
    
    def embed(self, input_data: Any, input_id: Optional[str] = None) -> Dict[str, Any]:
        result = self.manifold.embed(input_data, input_id)
        
        if input_id:
            self.current_context[input_id] = result
        
        return result
    
    def perceive(self, query: Dict[str, Any], context: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        if context is None:
            context = list(self.current_context.values())
        
        return self.entanglement_field.perceive_field(query, context)
    
    def check_constraints(self, geometric_reprs: List[Dict[str, Any]]) -> Dict[str, Any]:
        consistency_score = self.entropy_balance.evaluate_self_consistency(geometric_reprs)
        return self.constraints.validate_all(geometric_reprs, consistency_score)
    
    def reconstruct(self, inputs: List[Any], input_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        return self.mapper.reconstruct(inputs, input_ids)
    
    def evaluate_self_consistency(self, geometric_reprs: List[Dict[str, Any]]) -> float:
        return self.entropy_balance.evaluate_self_consistency(geometric_reprs)
    
    def compute_entropy_balance(self, geometric_reprs: List[Dict[str, Any]]) -> float:
        return self.entropy_balance.compute_entropy_balance(geometric_reprs)
    
    def update_entropy_balance(self, eta: float):
        self.manifold.update_entropy_balance(eta)
    
    def add_context(self, context_data: Any, context_id: Optional[str] = None):
        if context_id is None:
            context_id = f"ctx_{len(self.current_context)}"
        
        geo_state = self.embed(context_data, context_id)
        self.current_context[context_id] = geo_state
        
        return geo_state
    
    def clear_context(self):
        self.current_context = {}
    
    def get_system_state(self) -> Dict[str, Any]:
        entropy_state = self.entropy_balance.get_entropy_state()
        
        return {
            'manifold_dimension': self.manifold.dim,
            'num_entities': len(self.manifold.coords) if self.manifold.coords is not None else 0,
            'current_context_size': len(self.current_context),
            'history_length': len(self.history),
            'self_consistency_score': self.constraints.get_self_consistency_score(),
            'entropy_balance': entropy_state,
            'avg_curvature': np.mean(self.manifold.R_bg) if self.manifold.R_bg is not None else 0.0,
            'crh_density': self.manifold.rho if self.manifold.rho is not None else 0.0,
            'rho_history_length': len(self.manifold.rho_history)
        }
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        geo_repr = self.embed(input_data)
        
        context_list = list(self.current_context.values())
        perception = self.perceive(geo_repr, context_list)
        
        all_reprs = [geo_repr] + context_list
        constraint_check = self.check_constraints(all_reprs)
        
        consistency_score = self.evaluate_self_consistency(all_reprs)
        
        result = {
            'geometric_repr': geo_repr,
            'perception': perception,
            'constraint_check': constraint_check,
            'consistency_score': consistency_score,
            'valid': constraint_check['valid'] and consistency_score >= self.constraints.coherence_threshold
        }
        
        self.history.append(result)
        
        return result
    
    def update_from_feedback(self, verified_fact: Dict[str, Any], confidence: float = 0.9):
        """
        动态 ρ 更新机制：LLM验证通过的事实 → 更新几何流形
        
        Args:
            verified_fact: Dict with keys: 'summary', 'type', 'confidence', 'region'
            confidence: 事实置信度
        """
        fact_with_confidence = verified_fact.copy()
        fact_with_confidence['confidence'] = confidence
        
        self.manifold.update_from_feedback(fact_with_confidence, self.learning_rate)
    
    def estimate_complexity(self, query: str) -> float:
        """
        估算问题复杂度（用于多尺度剪枝）
        
        Args:
            query: 用户查询文本
        
        Returns:
            complexity_score: 0~1 的复杂度分数
        """
        words = query.split()
        word_count = len(words)
        
        special_terms = ['why', 'how', 'explain', 'describe', 'prove', 'derive', 'analyze', 'complex']
        special_count = sum(1 for word in words if word.lower() in special_terms)
        
        complexity_from_length = min(word_count / 20, 1.0)
        complexity_from_terms = special_count / len(special_terms)
        
        return (complexity_from_length + complexity_from_terms) / 2
    
    def get_scaled_manifold(self, complexity_score: float):
        """
        根据问题复杂度动态选择流形尺度
        
        Args:
            complexity_score: 0~1 的复杂度分数
        
        Returns:
            对应尺度的简化流形视图
        """
        return self.manifold.get_scaled_manifold(complexity_score)

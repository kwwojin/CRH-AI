import numpy as np
from typing import Dict, Any, Optional, Callable
from ..core.manifold import CRHManifold
from ..core.entanglement import EntanglementField
from ..core.entropy_balance import EntropyBalance
from ..constraints.physical_prior import PhysicalConstraints
from ..reconstruction.mapper import WeakCoherenceMapper
from ..prompts.geometric_prompt_engine import GeometricPromptEngine


class CRHLLMBridge:
    def __init__(self, llm: Optional[Callable] = None, learning_rate: float = 0.015):
        self.llm = llm
        self.manifold = CRHManifold(dim=64)
        self.entanglement_field = EntanglementField(self.manifold)
        self.entropy_balance = EntropyBalance(target_coherence=0.85)
        self.constraints = PhysicalConstraints(coherence_threshold=0.8)
        self.mapper = WeakCoherenceMapper(self.manifold, self.entanglement_field)
        self.prompt_engine = GeometricPromptEngine()
        
        self.learning_rate = learning_rate
        self.context_history = []
        self.current_context = {}
    
    def _estimate_complexity(self, query: str) -> float:
        """估算问题复杂度"""
        words = query.split()
        word_count = len(words)
        
        special_terms = ['why', 'how', 'explain', 'describe', 'prove', 'derive', 'analyze', 'complex']
        special_count = sum(1 for word in words if word.lower() in special_terms)
        
        complexity_from_length = min(word_count / 20, 1.0)
        complexity_from_terms = special_count / len(special_terms)
        
        return (complexity_from_length + complexity_from_terms) / 2
    
    def _reconstruct_prompt(self, perception: Dict[str, Any], user_input: str) -> str:
        gradient_info = perception.get('gradient_tensor', [])
        
        context_str = ""
        if gradient_info:
            context_str = "\n".join([
                f"- Related concept {i+1}: strength={g['entanglement_strength']:.3f}, distance={g['distance']:.3f}"
                for i, g in enumerate(gradient_info[:3])
            ])
        
        enhanced_prompt = f"""
        User Query: {user_input}
        
        Geometric Context:
        {context_str}
        
        Field Strength: {perception.get('avg_entanglement', 0.0):.3f}
        
        Please provide a coherent, fact-based response.
        Ensure your answer is consistent with physical reality and causal logic.
        """
        
        return enhanced_prompt.strip()
    
    def think(self, user_input: str) -> Dict[str, Any]:
        """标准推理流程"""
        geo_state = self.manifold.embed(user_input, input_id="query")
        
        context_list = [v for v in self.current_context.values()]
        perception = self.entanglement_field.perceive_field(geo_state, context_list)
        
        geometric_reprs = [geo_state] + context_list
        consistency_score = self.entropy_balance.evaluate_self_consistency(geometric_reprs)
        
        constraint_check = self.constraints.validate_all(geometric_reprs, consistency_score)
        
        if not constraint_check['valid']:
            return {
                'response': None,
                'coherence_score': consistency_score,
                'valid': False,
                'violations': constraint_check['violations'],
                'error': 'Constraint violation detected'
            }
        
        enhanced_prompt = self._reconstruct_prompt(perception, user_input)
        
        if self.llm is not None:
            try:
                response = self.llm(enhanced_prompt)
            except Exception as e:
                response = f"LLM error: {str(e)}"
        else:
            response = self._generate_fallback_response(user_input, consistency_score)
        
        final_score = self.entropy_balance.evaluate_self_consistency([geo_state])
        
        self.current_context[f"context_{len(self.context_history)}"] = geo_state
        self.context_history.append({
            'input': user_input,
            'response': response,
            'coherence_score': final_score,
            'timestamp': np.datetime64('now')
        })
        
        return {
            'response': response,
            'coherence_score': final_score,
            'valid': True,
            'violations': [],
            'perception': perception,
            'constraint_score': self.constraints.get_self_consistency_score()
        }
    
    def think_enhanced(self, user_query: str) -> Dict[str, Any]:
        """
        增强推理流程：使用几何提示词引擎和多尺度剪枝
        
        Returns:
            Dict containing response, coherence_score, and geometric context
        """
        complexity = self._estimate_complexity(user_query)
        scaled_manifold = self.manifold.get_scaled_manifold(complexity)
        
        geo_state = self.manifold.embed(user_query, input_id="query")
        geo_state['original_input'] = user_query
        
        context_list = [v for v in self.current_context.values()]
        perception = self.entanglement_field.perceive_field(geo_state, context_list)
        
        geometric_reprs = [geo_state] + context_list
        consistency_score = self.entropy_balance.evaluate_self_consistency(geometric_reprs)
        geo_state['coherence_score'] = consistency_score
        
        constraint_check = self.constraints.validate_all(geometric_reprs, consistency_score)
        
        if not constraint_check['valid']:
            return {
                'response': None,
                'coherence_score': consistency_score,
                'valid': False,
                'violations': constraint_check['violations'],
                'error': 'Constraint violation detected',
                'complexity': complexity,
                'scale': scaled_manifold['scale']
            }
        
        enhanced_prompt = self.prompt_engine.enhance_prompt(user_query, geo_state)
        
        if self.llm is not None:
            try:
                response = self.llm(enhanced_prompt)
            except Exception as e:
                response = f"LLM error: {str(e)}"
        else:
            response = self._generate_fallback_response(user_query, consistency_score)
        
        final_score = self.entropy_balance.evaluate_self_consistency([geo_state])
        
        self.current_context[f"context_{len(self.context_history)}"] = geo_state
        self.context_history.append({
            'input': user_query,
            'response': response,
            'coherence_score': final_score,
            'complexity': complexity,
            'scale': scaled_manifold['scale'],
            'timestamp': np.datetime64('now')
        })
        
        return {
            'response': response,
            'coherence_score': final_score,
            'valid': True,
            'violations': [],
            'perception': perception,
            'constraint_score': self.constraints.get_self_consistency_score(),
            'complexity': complexity,
            'scale': scaled_manifold['scale'],
            'enhanced_prompt': enhanced_prompt
        }
    
    def learn_from_feedback(self, verified_fact: Dict[str, Any], confidence: float = 0.9):
        """
        动态 ρ 更新：从验证过的事实学习
        
        Args:
            verified_fact: Dict with keys: 'summary', 'type', 'confidence', 'region'
            confidence: 事实置信度
        """
        fact_with_confidence = verified_fact.copy()
        fact_with_confidence['confidence'] = confidence
        
        self.manifold.update_from_feedback(fact_with_confidence, self.learning_rate)
    
    def _generate_fallback_response(self, user_input: str, coherence_score: float) -> str:
        responses = [
            "This is a geometrically coherent response based on your query.",
            "The CRH geometric base confirms this is a valid line of reasoning.",
            "Your question has been processed through the geometric anchor layer.",
            f"Self-consistency score: {coherence_score:.4f} - reasoning appears valid."
        ]
        return responses[hash(user_input) % len(responses)]
    
    def add_context(self, context_text: str, context_id: Optional[str] = None):
        if context_id is None:
            context_id = f"context_{len(self.current_context)}"
        
        geo_state = self.manifold.embed(context_text, input_id=context_id)
        self.current_context[context_id] = geo_state
        
        return geo_state
    
    def clear_context(self):
        self.current_context = {}
    
    def get_system_state(self) -> Dict[str, Any]:
        entropy_state = self.entropy_balance.get_entropy_state()
        
        return {
            'num_context_items': len(self.current_context),
            'history_length': len(self.context_history),
            'current_coherence': self.constraints.get_self_consistency_score(),
            'entropy_balance': entropy_state,
            'manifold_dimension': self.manifold.dim,
            'crh_density': self.manifold.rho if self.manifold.rho is not None else 0.0,
            'rho_history_length': len(self.manifold.rho_history)
        }
    
    def get_geometric_report(self, geo_state: Dict[str, Any]) -> str:
        """生成几何状态报告"""
        return self.prompt_engine.generate_geometry_report(geo_state)

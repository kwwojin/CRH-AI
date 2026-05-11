import numpy as np
from typing import Dict, Any, List, Optional, Union
from .layers.geometric_anchor import GeometricAnchorLayer
from .layers.entanglement_gradient import EntanglementGradientLayer
from .layers.physical_constraint import PhysicalConstraintLayer
from .layers.coherence_reconstruction import CoherenceReconstructionLayer


class CRHCoherenceBase:
    def __init__(self, embedding_dim: int = 5, coherence_threshold: float = 0.8):
        self.geometric_anchor = GeometricAnchorLayer(embedding_dim=embedding_dim)
        self.entanglement_gradient = EntanglementGradientLayer()
        self.physical_constraint = PhysicalConstraintLayer(coherence_threshold=coherence_threshold)
        self.coherence_reconstruction = CoherenceReconstructionLayer(
            anchor_layer=self.geometric_anchor,
            gradient_layer=self.entanglement_gradient
        )
        
        self.history = []
        self.current_context = {}
    
    def process_input(self, inputs: Union[Any, List[Any]], input_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        if input_ids is None:
            input_ids = [f"input_{i}_{np.random.randint(1000)}" for i in range(len(inputs))]
        
        reconstruction_result = self.coherence_reconstruction.reconstruct(inputs, input_ids)
        
        geometric_anchors = reconstruction_result['geometric_anchors']
        global_coherence = reconstruction_result['global_coherence_score']
        
        constraint_result = self.physical_constraint.check_constraint(geometric_anchors, global_coherence)
        
        output_decision = 'accept' if constraint_result['compliant'] and global_coherence >= self.physical_constraint.coherence_threshold else 'reject'
        
        result = {
            'stage': 'processing',
            'input_ids': input_ids,
            'geometric_anchors': geometric_anchors,
            'constraint_check': constraint_result,
            'coherence_scores': {
                'global': global_coherence,
                'local': reconstruction_result['local_coherence_score']
            },
            'output_decision': output_decision,
            'reconstruction_details': reconstruction_result['reconstruction_details']
        }
        
        if output_decision == 'accept':
            self.current_context.update({id_: geo for id_, geo in zip(input_ids, geometric_anchors)})
        
        self.history.append(result)
        
        return result
    
    def reason(self, query: Any, context_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        query_id = f"query_{np.random.randint(10000)}"
        
        query_features = self.coherence_reconstruction._extract_features(query)
        query_geometry = self.geometric_anchor.embed_to_geometry(query_features, query_id)
        query_geometry['id'] = query_id
        
        context_geometries = []
        if context_ids:
            for ctx_id in context_ids:
                ctx_geo = self.geometric_anchor.get_entity_geometry(ctx_id)
                if ctx_geo:
                    ctx_geo['id'] = ctx_id
                    context_geometries.append(ctx_geo)
        else:
            for ctx_id, ctx_geo in self.current_context.items():
                ctx_geo['id'] = ctx_id
                context_geometries.append(ctx_geo)
        
        gradient_result = self.entanglement_gradient.perceive_gradient(query_geometry, context_geometries)
        
        all_geometries = [query_geometry] + context_geometries
        coherence_result = self.entanglement_gradient.compute_coherence_score(all_geometries)
        
        constraint_result = self.physical_constraint.check_constraint(all_geometries, coherence_result['global_coherence'])
        
        reasoning_result = {
            'query_id': query_id,
            'query_geometry': query_geometry,
            'context_count': len(context_geometries),
            'gradient_perception': gradient_result,
            'coherence_scores': coherence_result,
            'constraint_check': constraint_result,
            'reasoning_quality': 'high' if constraint_result['compliant'] and coherence_result['global_coherence'] >= 0.8 else 'medium' if coherence_result['global_coherence'] >= 0.6 else 'low'
        }
        
        self.history.append({
            'stage': 'reasoning',
            'result': reasoning_result
        })
        
        return reasoning_result
    
    def generate(self, prompt: str, max_length: int = 100) -> Dict[str, Any]:
        processing_result = self.process_input(prompt)
        
        if processing_result['output_decision'] != 'accept':
            return {
                'success': False,
                'error': 'Input failed constraint check',
                'constraint_result': processing_result['constraint_check']
            }
        
        geometric_anchors = processing_result['geometric_anchors']
        
        generated_tokens = []
        current_context = geometric_anchors
        
        for _ in range(max_length):
            if not current_context:
                break
            
            avg_curvature = np.mean([g['curvature'] for g in current_context])
            avg_phase = np.mean([g['phase'] for g in current_context])
            
            token = self._curvature_to_token(avg_curvature, avg_phase)
            generated_tokens.append(token)
            
            if token == '<END>':
                break
            
            new_geo = {
                'coordinates': current_context[0]['coordinates'] + np.random.randn(len(current_context[0]['coordinates'])) * 0.1,
                'curvature': avg_curvature * 0.95,
                'phase': (avg_phase + 0.1) % (2 * np.pi),
                'topological_charge': current_context[0]['topological_charge']
            }
            current_context = [new_geo]
        
        return {
            'success': True,
            'generated_text': ' '.join(generated_tokens),
            'geometric_anchors': geometric_anchors,
            'coherence_score': processing_result['coherence_scores']['global'],
            'constraint_check': processing_result['constraint_check']
        }
    
    def _curvature_to_token(self, curvature: float, phase: float) -> str:
        if curvature < 0.3:
            return ['the', 'and', 'is', 'of', 'a'][int(phase * 5) % 5]
        elif curvature < 0.6:
            return ['thought', 'idea', 'concept', 'knowledge', 'understanding'][int(phase * 5) % 5]
        elif curvature < 0.9:
            return ['reason', 'logic', 'truth', 'fact', 'evidence'][int(phase * 5) % 5]
        else:
            return '<END>'
    
    def update_entropy_balance(self, eta: float):
        self.geometric_anchor.update_curvature_field(eta)
    
    def get_system_state(self) -> Dict[str, Any]:
        return {
            'entities_count': len(self.geometric_anchor.coordinates),
            'current_context_size': len(self.current_context),
            'average_coherence': np.mean([h.get('coherence_scores', {}).get('global', 0) for h in self.history]) if self.history else 0,
            'total_violations': sum(h.get('constraint_check', {}).get('num_violations', 0) for h in self.history),
            'field_state': self.entanglement_gradient.get_field_state(),
            'processing_history_count': len(self.history)
        }
    
    def reset_context(self):
        self.current_context = {}
        self.geometric_anchor = GeometricAnchorLayer(embedding_dim=self.geometric_anchor.embedding_dim)
        self.entanglement_gradient = EntanglementGradientLayer()
        self.history = []

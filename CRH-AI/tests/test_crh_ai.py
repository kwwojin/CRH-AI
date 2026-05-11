import pytest
import numpy as np
from src.crh_ai.layers.geometric_anchor import GeometricAnchorLayer
from src.crh_ai.layers.entanglement_gradient import EntanglementGradientLayer
from src.crh_ai.layers.physical_constraint import PhysicalConstraintLayer
from src.crh_ai.layers.coherence_reconstruction import CoherenceReconstructionLayer
from src.crh_ai.crh_core import CRHCoherenceBase


class TestGeometricAnchorLayer:
    def test_initialization(self):
        layer = GeometricAnchorLayer(embedding_dim=5)
        assert layer.embedding_dim == 5
        assert layer.R0 == 1.0
        assert layer.alpha == 0.1
        assert layer.rs == 0.5
    
    def test_embed_to_geometry(self):
        layer = GeometricAnchorLayer()
        result = layer.embed_to_geometry("Hello World", "test1")
        assert 'coordinates' in result
        assert 'curvature' in result
        assert 'phase' in result
        assert 'topological_charge' in result
        assert len(result['coordinates']) == 5
    
    def test_compute_geodesic(self):
        layer = GeometricAnchorLayer()
        layer.embed_to_geometry("First", "id1")
        layer.embed_to_geometry("Second", "id2")
        result = layer.compute_geodesic("id1", "id2")
        assert 'distance' in result
        assert 'path' in result
        assert result['distance'] >= 0
    
    def test_update_curvature_field(self):
        layer = GeometricAnchorLayer()
        layer.embed_to_geometry("Test", "test_id")
        initial_curvature = layer.curvatures["test_id"]
        layer.update_curvature_field(eta=2.0)
        assert layer.curvatures["test_id"] != initial_curvature


class TestEntanglementGradientLayer:
    def test_initialization(self):
        layer = EntanglementGradientLayer()
        assert layer.E0 == 1.0
        assert layer.beta == 0.5
    
    def test_perceive_gradient(self):
        layer = EntanglementGradientLayer()
        query = {'coordinates': np.array([0, 0, 0, 0, 0]), 'curvature': 1.0}
        context = [
            {'coordinates': np.array([1, 0, 0, 0, 0]), 'curvature': 0.8},
            {'coordinates': np.array([10, 0, 0, 0, 0]), 'curvature': 0.5}
        ]
        result = layer.perceive_gradient(query, context)
        assert 'gradient_tensor' in result
        assert len(result['gradient_tensor']) == 2
        assert result['gradient_tensor'][0]['entanglement_strength'] > result['gradient_tensor'][1]['entanglement_strength']
    
    def test_compute_coherence_score(self):
        layer = EntanglementGradientLayer()
        geometries = [
            {'coordinates': np.array([0, 0, 0, 0, 0]), 'curvature': 0.5},
            {'coordinates': np.array([1, 1, 0, 0, 0]), 'curvature': 0.6},
            {'coordinates': np.array([2, 2, 0, 0, 0]), 'curvature': 0.7}
        ]
        result = layer.compute_coherence_score(geometries)
        assert 'global_coherence' in result
        assert 'local_coherence' in result
        assert 0 <= result['global_coherence'] <= 1


class TestPhysicalConstraintLayer:
    def test_initialization(self):
        layer = PhysicalConstraintLayer()
        assert layer.coherence_threshold == 0.8
        assert layer.collision_threshold == 0.1
    
    def test_check_constraint_no_violations(self):
        layer = PhysicalConstraintLayer()
        geometries = [
            {'coordinates': np.array([0, 0, 0, 0, 0]), 'phase': 0.1, 'curvature': 0.5, 'topological_charge': 0.8},
            {'coordinates': np.array([1, 1, 1, 1, 1]), 'phase': 0.5, 'curvature': 0.6, 'topological_charge': 0.85},
            {'coordinates': np.array([2, 2, 2, 2, 2]), 'phase': 0.9, 'curvature': 0.7, 'topological_charge': 0.9}
        ]
        result = layer.check_constraint(geometries, global_coherence=0.9)
        assert result['compliant'] is True
        assert result['num_violations'] == 0
    
    def test_check_constraint_temporal_violation(self):
        layer = PhysicalConstraintLayer()
        geometries = [
            {'coordinates': np.array([0, 0, 0, 0, 0]), 'phase': 0.9, 'curvature': 0.5},
            {'coordinates': np.array([1, 0, 0, 0, 0]), 'phase': 0.1, 'curvature': 0.6}
        ]
        result = layer.check_constraint(geometries)
        assert result['compliant'] is False
        assert any(v['type'] == 'temporal' for v in result['violations'])


class TestCoherenceReconstructionLayer:
    def test_reconstruct(self):
        anchor = GeometricAnchorLayer()
        gradient = EntanglementGradientLayer()
        layer = CoherenceReconstructionLayer(anchor, gradient)
        
        inputs = ["Hello world", "Artificial intelligence", "Machine learning"]
        result = layer.reconstruct(inputs)
        
        assert 'high_coherence_representations' in result
        assert 'geometric_anchors' in result
        assert 'global_coherence_score' in result
        assert 0 <= result['global_coherence_score'] <= 1


class TestCRHCoherenceBase:
    def test_initialization(self):
        crh = CRHCoherenceBase()
        assert crh.geometric_anchor is not None
        assert crh.entanglement_gradient is not None
        assert crh.physical_constraint is not None
        assert crh.coherence_reconstruction is not None
    
    def test_process_input(self):
        crh = CRHCoherenceBase()
        result = crh.process_input("Hello CRH Geometry")
        assert 'input_ids' in result
        assert 'geometric_anchors' in result
        assert 'constraint_check' in result
        assert 'output_decision' in result
    
    def test_reason(self):
        crh = CRHCoherenceBase()
        crh.process_input("Context: The sky is blue")
        result = crh.reason("Why is the sky blue?")
        assert 'query_id' in result
        assert 'gradient_perception' in result
        assert 'reasoning_quality' in result
    
    def test_generate(self):
        crh = CRHCoherenceBase()
        result = crh.generate("The meaning of life is", max_length=10)
        assert 'success' in result
        if result['success']:
            assert 'generated_text' in result
        else:
            assert 'error' in result
    
    def test_get_system_state(self):
        crh = CRHCoherenceBase()
        crh.process_input("Test input")
        state = crh.get_system_state()
        assert 'entities_count' in state
        assert 'current_context_size' in state
        assert 'average_coherence' in state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

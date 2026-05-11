import pytest
import numpy as np
from src.crh_ai.core_base import CRHGeometricBase
from src.crh_ai.core.manifold import CRHManifold
from src.crh_ai.core.geometry import GeometricEngine
from src.crh_ai.core.entanglement import EntanglementField
from src.crh_ai.core.entropy_balance import EntropyBalance
from src.crh_ai.constraints.physical_prior import PhysicalConstraints
from src.crh_ai.integration.llm_bridge import CRHLLMBridge


class TestCRHManifold:
    def test_initialization(self):
        manifold = CRHManifold(dim=32)
        assert manifold.dim == 32
        assert manifold.rho is None
        assert manifold.coords is None
    
    def test_embed_text(self):
        manifold = CRHManifold(dim=32)
        result = manifold.embed("Hello World", "test1")
        assert 'coordinates' in result
        assert len(result['coordinates']) == 32
        assert 'curvature' in result
        assert 'crh_density' in result
    
    def test_get_curvature_at(self):
        manifold = CRHManifold(dim=32)
        manifold.embed("First", "id1")
        manifold.embed("Second", "id2")
        
        pos = manifold.coords[0]
        curvature = manifold.get_curvature_at(pos)
        assert curvature > 0
    
    def test_update_entropy_balance(self):
        manifold = CRHManifold(dim=32)
        manifold.embed("Test", "test_id")
        initial_curvature = manifold.R_bg[0]
        manifold.update_entropy_balance(eta=2.0)
        assert manifold.R_bg[0] != initial_curvature


class TestGeometricEngine:
    def test_initialization(self):
        engine = GeometricEngine(c=1.0)
        assert engine.c == 1.0
    
    def test_geometric_redshift(self):
        engine = GeometricEngine()
        redshift = engine.geometric_redshift(1.0, 0.5)
        assert redshift > 0
        assert redshift == np.exp(0.5) - 1
    
    def test_distance(self):
        engine = GeometricEngine()
        pos1 = np.array([0, 0, 0])
        pos2 = np.array([1, 1, 1])
        dist = engine.distance(pos1, pos2)
        assert dist == pytest.approx(np.sqrt(3))
    
    def test_causal_check(self):
        engine = GeometricEngine(c=1.0)
        event1 = {'t': 0, 'x': np.array([0, 0, 0])}
        event2 = {'t': 2, 'x': np.array([1, 0, 0])}
        assert engine.causal_check(event1, event2) == True
        
        event3 = {'t': 0.5, 'x': np.array([1, 0, 0])}
        assert engine.causal_check(event1, event3) == False


class TestEntanglementField:
    def test_compute_gradient(self):
        manifold = CRHManifold(dim=32)
        field = EntanglementField(manifold)
        
        entity1 = {'coordinates': np.array([0, 0, 0, 0, 0]), 'crh_density': 1.0}
        entity2 = {'coordinates': np.array([1, 0, 0, 0, 0]), 'crh_density': 1.0}
        
        strength = field.compute_gradient(entity1, entity2)
        assert 0 < strength <= 1.0
    
    def test_perceive_field(self):
        manifold = CRHManifold(dim=32)
        field = EntanglementField(manifold)
        
        query = {'coordinates': np.array([0, 0, 0, 0, 0])}
        context = [
            {'coordinates': np.array([1, 0, 0, 0, 0])},
            {'coordinates': np.array([10, 0, 0, 0, 0])}
        ]
        
        result = field.perceive_field(query, context)
        assert 'gradient_tensor' in result
        assert len(result['gradient_tensor']) == 2


class TestEntropyBalance:
    def test_evaluate_self_consistency(self):
        balance = EntropyBalance()
        
        reprs = [
            {'coordinates': np.array([0, 0, 0]), 'curvature': 0.5},
            {'coordinates': np.array([1, 1, 0]), 'curvature': 0.6}
        ]
        
        score = balance.evaluate_self_consistency(reprs)
        assert 0 <= score <= 1.0
    
    def test_compute_entropy_balance(self):
        balance = EntropyBalance()
        
        reprs = [
            {'coordinates': np.array([0, 0, 0]), 'curvature': 0.5},
            {'coordinates': np.array([1, 0, 0]), 'curvature': 0.5}
        ]
        
        eta = balance.compute_entropy_balance(reprs)
        assert 0.5 <= eta <= 1.5
    
    def test_get_entropy_state(self):
        balance = EntropyBalance()
        state = balance.get_entropy_state()
        assert 'current_eta' in state
        assert 'consistency_trend' in state


class TestPhysicalConstraints:
    def test_check_consistency(self):
        constraints = PhysicalConstraints()
        
        proposed_action = {
            'entities': [
                {'coordinates': np.array([0.5, 0.5, 0.5]), 'curvature': 0.5},
                {'coordinates': np.array([1.0, 1.0, 1.0]), 'curvature': 0.6}
            ],
            'consistency_score': 0.9
        }
        
        current_state = {'energy': 1.0, 'information': 2}
        valid, violations = constraints.check_consistency(proposed_action, current_state)
        
        assert valid == True
        assert len(violations) == 0
    
    def test_validate_all(self):
        constraints = PhysicalConstraints()
        
        reprs = [
            {'coordinates': np.array([0, 0, 0]), 'curvature': 0.5},
            {'coordinates': np.array([0.05, 0.05, 0.05]), 'curvature': 0.6}
        ]
        
        result = constraints.validate_all(reprs, 0.9)
        assert 'valid' in result
        assert 'self_consistency_score' in result


class TestCRHGeometricBase:
    def test_initialization(self):
        base = CRHGeometricBase(dim=64)
        assert base.manifold.dim == 64
        assert base.constraints.coherence_threshold == 0.8
    
    def test_embed(self):
        base = CRHGeometricBase()
        result = base.embed("Hello CRH")
        assert 'coordinates' in result
        assert 'curvature' in result
    
    def test_perceive(self):
        base = CRHGeometricBase()
        base.add_context("Context 1")
        base.add_context("Context 2")
        
        query = base.embed("Query")
        perception = base.perceive(query)
        
        assert 'gradient_tensor' in perception
        assert 'avg_entanglement' in perception
    
    def test_process(self):
        base = CRHGeometricBase()
        result = base.process("Test input")
        
        assert 'geometric_repr' in result
        assert 'perception' in result
        assert 'constraint_check' in result
        assert 'consistency_score' in result
        assert 'valid' in result
    
    def test_get_system_state(self):
        base = CRHGeometricBase()
        base.process("Test")
        state = base.get_system_state()
        
        assert 'manifold_dimension' in state
        assert 'num_entities' in state
        assert 'self_consistency_score' in state


class TestCRHLLMBridge:
    def test_initialization(self):
        bridge = CRHLLMBridge()
        assert bridge.manifold is not None
        assert bridge.constraints is not None
    
    def test_think_without_llm(self):
        bridge = CRHLLMBridge()
        result = bridge.think("What is AI?")
        
        assert 'response' in result
        assert 'coherence_score' in result
        assert 'valid' in result
    
    def test_add_context(self):
        bridge = CRHLLMBridge()
        geo_state = bridge.add_context("Context information")
        
        assert geo_state is not None
        assert 'coordinates' in geo_state
        assert len(bridge.current_context) == 1
    
    def test_get_system_state(self):
        bridge = CRHLLMBridge()
        bridge.think("Test query")
        state = bridge.get_system_state()
        
        assert 'num_context_items' in state
        assert 'history_length' in state
        assert 'current_coherence' in state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

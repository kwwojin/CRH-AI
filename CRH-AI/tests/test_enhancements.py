import pytest
import numpy as np
from src.crh_ai import CRHManifold, CRHGeometricBase, CRHLLMBridge, GeometricPromptEngine


class TestDynamicRhoUpdate:
    def test_update_from_feedback(self):
        manifold = CRHManifold(dim=32)
        manifold.embed("Initial data")
        
        initial_rho = manifold.rho
        
        verified_fact = {
            'summary': 'Earth is approximately spherical',
            'type': 'physical',
            'region': 'solar_system',
            'confidence': 0.95
        }
        
        manifold.update_from_feedback(verified_fact)
        
        assert manifold.rho is not None
        assert manifold.rho != initial_rho
        assert 0.1 <= manifold.rho <= 0.99
    
    def test_rho_history(self):
        manifold = CRHManifold(dim=32)
        manifold.embed("Test")
        
        for i in range(5):
            fact = {
                'summary': f'Fact {i}',
                'type': 'general',
                'region': 'test_region',
                'confidence': 0.8 + i * 0.02
            }
            manifold.update_from_feedback(fact)
        
        history = manifold.get_rho_history()
        assert len(history) == 5
        assert all('timestamp' in h for h in history)
        assert all('rho' in h for h in history)
    
    def test_different_fact_types(self):
        manifold = CRHManifold(dim=32)
        manifold.embed("Base")
        
        facts = [
            {'summary': 'Physical law', 'type': 'physical', 'region': 'physics', 'confidence': 0.9},
            {'summary': 'Causal relation', 'type': 'causal', 'region': 'causality', 'confidence': 0.9},
            {'summary': 'Math theorem', 'type': 'mathematical', 'region': 'math', 'confidence': 0.9},
            {'summary': 'Contradiction', 'type': 'contradiction', 'region': 'test', 'confidence': 0.9}
        ]
        
        for fact in facts:
            manifold.update_from_feedback(fact)
        
        assert len(manifold.rho_history) == 4


class TestGeometricPromptEngine:
    def test_enhance_prompt(self):
        engine = GeometricPromptEngine()
        
        geo_state = {
            'coordinates': np.array([0.5, 0.5, 0.5]),
            'curvature': 0.7,
            'crh_density': 0.75,
            'coherence_score': 0.927,
            'original_input': 'Describe black hole horizon'
        }
        
        enhanced = engine.enhance_prompt('Describe black hole horizon', geo_state)
        
        assert '[几何相干上下文]' in enhanced
        assert '对称性' in enhanced
        assert '曲率强度' in enhanced
        assert '纠缠梯度' in enhanced
        assert '拓扑结构' in enhanced
        assert '因果方向' in enhanced
        assert '0.927' in enhanced
    
    def test_generate_report(self):
        engine = GeometricPromptEngine()
        
        geo_state = {
            'coordinates': np.array([1.0, 2.0, 3.0]),
            'curvature': 0.6,
            'crh_density': 0.7,
            'coherence_score': 0.85,
            'dimensionality': 64,
            'id': 'test_id'
        }
        
        report = engine.generate_geometry_report(geo_state)
        
        assert '几何状态报告' in report
        assert '维度' in report
        assert '坐标范数' in report
        assert '局部曲率' in report


class TestMultiscalePruning:
    def test_get_scaled_manifold(self):
        manifold = CRHManifold(dim=64)
        manifold.embed("Data 1")
        manifold.embed("Data 2")
        manifold.embed("Data 3")
        
        scaled_low = manifold.get_scaled_manifold(0.2)
        assert scaled_low['scale'] == 1
        
        scaled_medium = manifold.get_scaled_manifold(0.5)
        assert scaled_medium['scale'] == 2
        
        scaled_high = manifold.get_scaled_manifold(0.7)
        assert scaled_high['scale'] == 4
        
        scaled_extreme = manifold.get_scaled_manifold(0.9)
        assert scaled_extreme['scale'] == 8
    
    def test_coarse_grained_view(self):
        manifold = CRHManifold(dim=16)
        manifold.embed("Test data")
        
        scaled = manifold.get_scaled_manifold(0.9)
        
        assert 'coords' in scaled
        assert 'R_bg' in scaled
        assert 'rho' in scaled
        assert 'scale' in scaled
        assert 'dim' in scaled
        assert scaled['dim'] == 2


class TestCRHGeometricBaseEnhancements:
    def test_update_from_feedback(self):
        base = CRHGeometricBase(dim=32, learning_rate=0.02)
        base.embed("Initial")
        
        fact = {
            'summary': 'Test fact',
            'type': 'physical',
            'region': 'test',
            'confidence': 0.9
        }
        
        base.update_from_feedback(fact)
        
        state = base.get_system_state()
        assert state['rho_history_length'] == 1
    
    def test_estimate_complexity(self):
        base = CRHGeometricBase(dim=32)
        
        simple = base.estimate_complexity("Hello")
        assert 0 <= simple <= 1.0
        
        medium = base.estimate_complexity("Explain why quantum mechanics is important")
        assert 0 <= medium <= 1.0
        
        complex_query = base.estimate_complexity("Prove Fermat's last theorem using advanced algebraic geometry")
        assert 0 <= complex_query <= 1.0
        
        empty = base.estimate_complexity("")
        assert empty == 0.0


class TestCRHLLMBridgeEnhancements:
    def test_think_enhanced(self):
        bridge = CRHLLMBridge()
        
        bridge.add_context("Physics is the study of matter and energy")
        bridge.add_context("Redshift is the displacement of spectral lines toward longer wavelengths")
        
        result = bridge.think_enhanced("Explain geometric redshift")
        
        assert 'response' in result
        assert 'coherence_score' in result
        assert 'complexity' in result
        assert 'scale' in result
        if result.get('valid'):
            assert 'enhanced_prompt' in result
            assert '[几何相干上下文]' in result['enhanced_prompt']
    
    def test_learn_from_feedback(self):
        bridge = CRHLLMBridge()
        
        fact = {
            'summary': 'Light travels at 3e8 m/s',
            'type': 'physical',
            'region': 'physics',
            'confidence': 0.95
        }
        
        bridge.learn_from_feedback(fact)
        
        state = bridge.get_system_state()
        assert state['rho_history_length'] == 1
    
    def test_get_geometric_report(self):
        bridge = CRHLLMBridge()
        geo_state = bridge.add_context("Test context")
        
        report = bridge.get_geometric_report(geo_state)
        
        assert '几何状态报告' in report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

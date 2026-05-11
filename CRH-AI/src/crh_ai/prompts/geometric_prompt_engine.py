import numpy as np
from typing import Dict, Any, Optional


class GeometricPromptEngine:
    def __init__(self):
        self.symmetry_patterns = {
            'spherical': ['sphere', 'ball', 'round', 'planet', 'star'],
            'cylindrical': ['cylinder', 'tube', 'pipe', 'rod'],
            'planar': ['plane', 'flat', 'sheet', 'disk'],
            'toroidal': ['torus', 'donut', 'ring'],
            'fractal': ['fractal', 'self-similar', 'complex']
        }
        
        self.curvature_levels = [
            (0.0, 0.2, 'Very Low'),
            (0.2, 0.4, 'Low'),
            (0.4, 0.6, 'Medium'),
            (0.6, 0.8, 'Medium-High'),
            (0.8, 1.0, 'High'),
            (1.0, float('inf'), 'Extreme')
        ]
        
        self.entanglement_levels = [
            (0.0, 0.3, 'Weak-Global'),
            (0.3, 0.5, 'Weak-Local'),
            (0.5, 0.7, 'Medium'),
            (0.7, 0.85, 'Strong-Local'),
            (0.85, 1.0, 'Strong-Global')
        ]
        
        self.topology_types = {
            'simply_connected': 'Simply Connected',
            'toroidal': 'Toroidal',
            'hopfion': 'Hopfion-like',
            'fractal': 'Fractal',
            'disconnected': 'Disconnected'
        }
    
    def _detect_symmetry(self, geo_state: Dict[str, Any]) -> str:
        if 'coordinates' not in geo_state:
            return 'Unknown'
        
        coords = geo_state['coordinates']
        
        if len(coords) >= 3:
            r = np.linalg.norm(coords[:3])
            theta = np.arccos(coords[2] / r) if r > 0 else 0
            phi = np.arctan2(coords[1], coords[0])
            
            if np.abs(theta - np.pi/4) < 0.3 and np.abs(phi - np.pi/4) < 0.3:
                return 'Spherically Symmetric'
        
        if 'original_input' in geo_state:
            text = geo_state['original_input'].lower()
            for pattern, keywords in self.symmetry_patterns.items():
                if any(keyword in text for keyword in keywords):
                    return pattern.capitalize() + ' Symmetric'
        
        return 'Asymmetric'
    
    def _get_curvature_label(self, R_bg: Optional[float]) -> str:
        if R_bg is None:
            return 'Unknown'
        
        for low, high, label in self.curvature_levels:
            if low <= R_bg < high:
                return label
        
        return 'Unknown'
    
    def _get_entanglement_label(self, geo_state: Dict[str, Any]) -> str:
        if 'crh_density' not in geo_state:
            return 'Unknown'
        
        density = geo_state['crh_density']
        
        for low, high, label in self.entanglement_levels:
            if low <= density < high:
                return label
        
        return 'Unknown'
    
    def _get_topology_label(self, geo_state: Dict[str, Any]) -> str:
        if 'curvature' not in geo_state:
            return 'Unknown'
        
        curvature = geo_state['curvature']
        
        if curvature < 0.3:
            return self.topology_types['simply_connected']
        elif curvature < 0.6:
            return self.topology_types['toroidal']
        elif curvature < 0.8:
            return self.topology_types['hopfion']
        else:
            return self.topology_types['fractal']
    
    def _get_causal_direction(self, geo_state: Dict[str, Any]) -> str:
        if 'phase' in geo_state:
            phase = geo_state['phase']
            if phase < np.pi:
                return 'Forward'
            else:
                return 'Backward'
        
        if 'original_input' in geo_state:
            text = geo_state['original_input'].lower()
            if 'before' in text or 'cause' in text or 'past' in text:
                return 'Backward'
        
        return 'Forward'
    
    def enhance_prompt(self, user_query: str, geo_state: Dict[str, Any]) -> str:
        """生成带几何描述子的增强提示"""
        
        geo_desc = {
            "symmetry": self._detect_symmetry(geo_state),
            "curvature_level": self._get_curvature_label(geo_state.get('curvature')),
            "entanglement_strength": self._get_entanglement_label(geo_state),
            "topology_type": self._get_topology_label(geo_state),
            "causal_gradient": self._get_causal_direction(geo_state),
            "self_consistency": f"{geo_state.get('coherence_score', 0.0):.3f}"
        }
        
        enhanced_prompt = f"""
[几何相干上下文]
- 对称性: {geo_desc['symmetry']}
- 曲率强度: {geo_desc['curvature_level']}
- 纠缠梯度: {geo_desc['entanglement_strength']}
- 拓扑结构: {geo_desc['topology_type']}
- 因果方向: {geo_desc['causal_gradient']}
- 当前自洽性: {geo_desc['self_consistency']}

用户查询: {user_query}

请在以上几何约束下进行推理，确保输出与物理几何现实高度一致。
如果推理过程中发现矛盾，请指出并基于几何约束进行修正。
"""
        
        return enhanced_prompt.strip()
    
    def generate_geometry_report(self, geo_state: Dict[str, Any]) -> str:
        """生成详细的几何状态报告"""
        
        report = f"""
几何状态报告
============

[基本信息]
维度: {geo_state.get('dimensionality', 'Unknown')}
标识符: {geo_state.get('id', 'None')}

[坐标信息]
坐标向量维度: {len(geo_state.get('coordinates', []))}
坐标范数: {np.linalg.norm(geo_state.get('coordinates', [])):.4f}

[曲率信息]
局部曲率: {geo_state.get('curvature', 'Unknown')}
曲率级别: {self._get_curvature_label(geo_state.get('curvature'))}

[纠缠信息]
CRH密度: {geo_state.get('crh_density', 'Unknown')}
纠缠强度: {self._get_entanglement_label(geo_state)}

[拓扑信息]
拓扑类型: {self._get_topology_label(geo_state)}
对称性: {self._detect_symmetry(geo_state)}

[因果信息]
相位: {geo_state.get('phase', 'Unknown')}
因果方向: {self._get_causal_direction(geo_state)}

[自洽性]
自洽性分数: {geo_state.get('coherence_score', 'Unknown')}
"""
        
        return report.strip()

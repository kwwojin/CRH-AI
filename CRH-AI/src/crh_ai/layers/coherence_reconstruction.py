import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
from typing import Dict, Any, List, Tuple, Optional, Union
from .geometric_anchor import GeometricAnchorLayer
from .entanglement_gradient import EntanglementGradientLayer


class CoherenceReconstructionLayer:
    def __init__(self, anchor_layer: GeometricAnchorLayer, gradient_layer: EntanglementGradientLayer):
        self.anchor_layer = anchor_layer
        self.gradient_layer = gradient_layer
        
        self.reconstructed_history = []
    
    def _extract_features(self, input_data: Any) -> np.ndarray:
        if isinstance(input_data, str):
            return self._text_feature_extraction(input_data)
        elif isinstance(input_data, np.ndarray):
            return input_data.flatten()[:200]
        elif isinstance(input_data, dict):
            return self._dict_to_features(input_data)
        else:
            return np.array([hash(str(input_data)) % 1000 / 1000])
    
    def _text_feature_extraction(self, text: str) -> np.ndarray:
        words = text.lower().split()
        char_hist = np.zeros(26)
        for char in text.lower():
            if 'a' <= char <= 'z':
                char_hist[ord(char) - ord('a')] += 1
        
        word_len_dist = np.zeros(10)
        for word in words:
            idx = min(len(word) - 1, 9)
            word_len_dist[idx] += 1
        
        features = np.concatenate([
            char_hist / np.sum(char_hist) if np.sum(char_hist) > 0 else char_hist,
            word_len_dist / np.sum(word_len_dist) if np.sum(word_len_dist) > 0 else word_len_dist,
            np.array([len(text), len(words), len(set(words)) / max(len(words), 1)])
        ])
        
        return features
    
    def _dict_to_features(self, data: Dict[str, Any]) -> np.ndarray:
        features = []
        for key, value in sorted(data.items()):
            if isinstance(value, (int, float)):
                features.append(value)
            elif isinstance(value, str):
                features.append(hash(value) % 1000 / 1000)
        return np.array(features) if features else np.array([0.0])
    
    def _map_to_geometry(self, features: np.ndarray, input_id: str) -> Dict[str, Any]:
        return self.anchor_layer.embed_to_geometry(features, input_id)
    
    def _cluster_and_complement(self, geometric_reprs: List[Dict[str, Any]], threshold: float = 0.5) -> List[Dict[str, Any]]:
        if len(geometric_reprs) < 2:
            return geometric_reprs
        
        coords = np.array([g['coordinates'] for g in geometric_reprs])
        distances = cdist(coords, coords)
        
        linked = linkage(distances, 'ward')
        clusters = fcluster(linked, threshold, criterion='distance')
        
        cluster_groups = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(geometric_reprs[i])
        
        reconstructed = []
        for cluster_id, group in cluster_groups.items():
            if len(group) == 1:
                reconstructed.append(group[0])
            else:
                avg_coord = np.mean([g['coordinates'] for g in group], axis=0)
                avg_curvature = np.mean([g['curvature'] for g in group])
                avg_phase = np.mean([g['phase'] for g in group])
                avg_charge = np.mean([g['topological_charge'] for g in group])
                
                reconstructed.append({
                    'coordinates': avg_coord,
                    'curvature': avg_curvature,
                    'phase': avg_phase,
                    'topological_charge': avg_charge,
                    'cluster_size': len(group),
                    'original_ids': [g.get('id', i) for i, g in enumerate(group)]
                })
        
        return reconstructed
    
    def _prioritize_high_entanglement(self, geometric_reprs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(geometric_reprs) < 2:
            return geometric_reprs
        
        query_repr = geometric_reprs[0]
        context_reprs = geometric_reprs[1:]
        
        gradient_result = self.gradient_layer.perceive_gradient(query_repr, context_reprs)
        gradient_tensor = gradient_result['gradient_tensor']
        
        sorted_indices = [g['index'] + 1 for g in gradient_tensor]
        
        sorted_reprs = [geometric_reprs[0]] + [geometric_reprs[i] for i in sorted_indices]
        
        return sorted_reprs[:min(len(sorted_reprs), 10)]
    
    def reconstruct(self, weak_coherence_inputs: List[Any], input_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        if input_ids is None:
            input_ids = [f"input_{i}" for i in range(len(weak_coherence_inputs))]
        
        geometric_reprs = []
        for idx, input_data in enumerate(weak_coherence_inputs):
            features = self._extract_features(input_data)
            geo_repr = self._map_to_geometry(features, input_ids[idx])
            geo_repr['id'] = input_ids[idx]
            geo_repr['original_input'] = input_data
            geometric_reprs.append(geo_repr)
        
        geometric_reprs = self._prioritize_high_entanglement(geometric_reprs)
        
        geometric_reprs = self._cluster_and_complement(geometric_reprs)
        
        coherence_result = self.gradient_layer.compute_coherence_score(geometric_reprs)
        global_coherence = coherence_result['global_coherence']
        
        symbol_reprs = [self.geometric_to_symbol(geo) for geo in geometric_reprs]
        
        reconstruction_result = {
            'high_coherence_representations': symbol_reprs,
            'geometric_anchors': geometric_reprs,
            'global_coherence_score': global_coherence,
            'local_coherence_score': coherence_result['local_coherence'],
            'num_reconstructed': len(geometric_reprs),
            'reconstruction_details': {
                'avg_distance': coherence_result['avg_distance'],
                'curvature_std': coherence_result['curvature_std']
            }
        }
        
        self.reconstructed_history.append(reconstruction_result)
        
        return reconstruction_result
    
    def geometric_to_symbol(self, geometric_repr: Dict[str, Any]) -> str:
        coord = geometric_repr['coordinates']
        curvature = geometric_repr['curvature']
        phase = geometric_repr['phase']
        charge = geometric_repr['topological_charge']
        
        coord_str = ",".join([f"{c:.3f}" for c in coord])
        symbol_parts = [
            f"COORD({coord_str})",
            f"CURV({curvature:.4f})",
            f"PHSE({phase:.4f})",
            f"CHRG({charge:.4f})"
        ]
        
        return "|".join(symbol_parts)
    
    def symbol_to_geometric(self, symbol: str) -> Optional[Dict[str, Any]]:
        parts = symbol.split("|")
        result = {}
        
        for part in parts:
            if part.startswith("COORD("):
                coords = part.replace("COORD(", "").replace(")", "").split(",")
                result['coordinates'] = np.array([float(c) for c in coords])
            elif part.startswith("CURV("):
                result['curvature'] = float(part.replace("CURV(", "").replace(")", ""))
            elif part.startswith("PHSE("):
                result['phase'] = float(part.replace("PHSE(", "").replace(")", ""))
            elif part.startswith("CHRG("):
                result['topological_charge'] = float(part.replace("CHRG(", "").replace(")", ""))
        
        if 'coordinates' not in result:
            return None
        
        return result
    
    def get_reconstruction_history(self) -> List[Dict[str, Any]]:
        return self.reconstructed_history

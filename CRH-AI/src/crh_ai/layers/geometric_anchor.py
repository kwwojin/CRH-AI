import numpy as np
from scipy.spatial.distance import cdist
from sklearn.manifold import MDS
from typing import Tuple, Dict, Any, Optional


class GeometricAnchorLayer:
    def __init__(self, embedding_dim: int = 5, R0: float = 1.0, alpha: float = 0.1, rs: float = 0.5):
        self.embedding_dim = embedding_dim
        self.R0 = R0
        self.alpha = alpha
        self.rs = rs
        
        self.geometric_field = None
        self.coordinates = {}
        self.curvatures = {}
        self.phases = {}
        self.topological_charges = {}
        
        self.mds = MDS(n_components=embedding_dim, dissimilarity='precomputed', random_state=42)
        self.mds_embedding = None
    
    def _compute_crh_density(self, features: np.ndarray) -> np.ndarray:
        normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
        similarity = np.dot(normalized, normalized.T)
        density = np.mean(similarity, axis=1)
        return density
    
    def _compute_background_curvature(self, distance: float, eta: float = 1.0) -> float:
        return (self.R0 * eta) / (1 + self.alpha * distance) * (1 + self.rs / max(distance, 1e-6))
    
    def _compute_geodesic_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        diff = x2 - x1
        metric = np.eye(len(x1))
        return np.sqrt(np.dot(diff.T, np.dot(metric, diff)))
    
    def _compute_geometric_redshift(self, curvature: float) -> float:
        return np.exp(curvature) - 1
    
    def _compute_topological_charge(self, point: np.ndarray) -> float:
        normalized_point = point / np.linalg.norm(point) if np.linalg.norm(point) > 0 else point
        return np.sum(normalized_point ** 2)
    
    def embed_to_geometry(self, input_data: Any, input_id: str) -> Dict[str, Any]:
        if isinstance(input_data, str):
            features = self._text_to_features(input_data)
        elif isinstance(input_data, np.ndarray):
            features = input_data
        else:
            features = self._multimodal_to_features(input_data)
        
        crh_density = self._compute_crh_density(features.reshape(1, -1))[0]
        
        if self.geometric_field is None:
            self.geometric_field = features.reshape(1, -1)
        else:
            self.geometric_field = np.vstack([self.geometric_field, features.reshape(1, -1)])
        
        if len(self.geometric_field) >= 2:
            distances = cdist(self.geometric_field, self.geometric_field, 'euclidean')
            self.mds_embedding = self.mds.fit_transform(distances)
            coord = self.mds_embedding[-1]
        else:
            coord = np.random.randn(self.embedding_dim) * 0.1
        
        if self.mds_embedding is not None and len(self.mds_embedding) > 1:
            avg_distance = np.mean(cdist(coord.reshape(1, -1), self.mds_embedding[:-1], 'euclidean'))
        else:
            avg_distance = 0.5
        
        curvature = self._compute_background_curvature(avg_distance, eta=crh_density)
        phase = np.mod(np.sum(coord) * 2 * np.pi, 2 * np.pi)
        topological_charge = self._compute_topological_charge(coord)
        
        self.coordinates[input_id] = coord
        self.curvatures[input_id] = curvature
        self.phases[input_id] = phase
        self.topological_charges[input_id] = topological_charge
        
        return {
            'coordinates': coord,
            'curvature': curvature,
            'phase': phase,
            'topological_charge': topological_charge,
            'crh_density': crh_density
        }
    
    def compute_geodesic(self, id1: str, id2: str) -> Dict[str, Any]:
        if id1 not in self.coordinates or id2 not in self.coordinates:
            raise ValueError(f"One or both IDs not found: {id1}, {id2}")
        
        x1 = self.coordinates[id1]
        x2 = self.coordinates[id2]
        distance = self._compute_geodesic_distance(x1, x2)
        
        path = np.linspace(x1, x2, num=10)
        
        return {
            'distance': distance,
            'path': path,
            'start': x1,
            'end': x2
        }
    
    def update_curvature_field(self, eta: float = 1.0):
        for input_id in self.coordinates:
            coord = self.coordinates[input_id]
            avg_distance = np.mean(cdist(coord.reshape(1, -1), 
                                        np.array(list(self.coordinates.values())), 
                                        'euclidean'))
            self.curvatures[input_id] = self._compute_background_curvature(avg_distance, eta)
    
    def get_entity_geometry(self, input_id: str) -> Optional[Dict[str, Any]]:
        if input_id not in self.coordinates:
            return None
        return {
            'coordinates': self.coordinates[input_id],
            'curvature': self.curvatures[input_id],
            'phase': self.phases[input_id],
            'topological_charge': self.topological_charges[input_id]
        }
    
    def _text_to_features(self, text: str) -> np.ndarray:
        char_counts = np.zeros(26)
        for char in text.lower():
            if 'a' <= char <= 'z':
                char_counts[ord(char) - ord('a')] += 1
        if np.sum(char_counts) > 0:
            char_counts = char_counts / np.sum(char_counts)
        word_count = len(text.split())
        return np.concatenate([char_counts, np.array([word_count, len(text)])])
    
    def _multimodal_to_features(self, data: Any) -> np.ndarray:
        if hasattr(data, 'shape'):
            arr = np.array(data)
            return arr.flatten()[:100] if arr.size > 100 else arr.flatten()
        return np.array([hash(str(data)) % 1000 / 1000])

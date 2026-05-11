import numpy as np
from sklearn.manifold import MDS
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
from typing import Dict, Any, Optional, List


class CRHManifold:
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.rho = None                    
        self.coords = None                 
        self.R_bg = None                   
        self.features = None               
        self.mds = MDS(n_components=dim, dissimilarity='precomputed', random_state=42, n_init=1)
        
        self.R0 = 1.0                      
        self.alpha = 0.1                   
        self.rs = 0.5                      
        
        self.rho_history = []              
        self.scaled_manifolds = {}         
    
    def _compute_crh_density(self, features: np.ndarray) -> np.ndarray:
        normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
        similarity = np.dot(normalized, normalized.T)
        density = np.mean(similarity, axis=1)
        return density
    
    def _compute_background_curvature(self, distance: float, eta: float = 1.0) -> float:
        return (self.R0 * eta) / (1 + self.alpha * distance) * (1 + self.rs / max(distance, 1e-6))
    
    def embed(self, input_data: Any, input_id: Optional[str] = None) -> Dict[str, Any]:
        if isinstance(input_data, str):
            features = self._text_to_features(input_data)
        elif isinstance(input_data, np.ndarray):
            features = input_data.flatten()[:self.dim]
        else:
            features = self._multimodal_to_features(input_data)
        
        features = features.reshape(1, -1)
        
        if self.features is None:
            self.features = features
        else:
            self.features = np.vstack([self.features, features])
        
        if len(self.features) >= 2:
            distances = cdist(self.features, self.features, 'euclidean')
            np.fill_diagonal(distances, 0)
            self.coords = self.mds.fit_transform(distances)
            coord = self.coords[-1]
        else:
            coord = np.random.randn(self.dim) * 0.1
            self.coords = coord.reshape(1, -1)
        
        crh_density = self._compute_crh_density(self.features)[-1]
        
        if self.coords is not None and len(self.coords) > 1:
            avg_distance = np.mean(cdist(coord.reshape(1, -1), self.coords[:-1], 'euclidean'))
        else:
            avg_distance = 0.5
        
        curvature = self._compute_background_curvature(avg_distance, eta=crh_density)
        
        if self.R_bg is None:
            self.R_bg = np.array([curvature])
        else:
            self.R_bg = np.append(self.R_bg, curvature)
        
        self.rho = crh_density
        
        return {
            'coordinates': coord,
            'curvature': curvature,
            'crh_density': crh_density,
            'id': input_id,
            'dimensionality': self.dim
        }
    
    def get_curvature_at(self, position: np.ndarray) -> float:
        if self.coords is None:
            return self.R0
        
        if len(self.coords) == 1:
            return self.R_bg[0] if self.R_bg is not None else self.R0
        
        distances = cdist(position.reshape(1, -1), self.coords)[0]
        nearest_idx = np.argmin(distances)
        
        if self.R_bg is not None and nearest_idx < len(self.R_bg):
            return self.R_bg[nearest_idx]
        
        return self._compute_background_curvature(distances[nearest_idx])
    
    def update_entropy_balance(self, eta: float):
        if self.coords is not None and len(self.coords) > 0:
            for i in range(len(self.coords)):
                if i < len(self.coords) - 1:
                    distances = cdist(self.coords[i].reshape(1, -1), self.coords[:i], 'euclidean')
                    avg_dist = np.mean(distances) if len(distances) > 0 else 0.5
                else:
                    avg_dist = 0.5
                self.R_bg[i] = self._compute_background_curvature(avg_dist, eta)
    
    def _fact_to_geometric_delta(self, verified_fact: Dict[str, Any]) -> float:
        fact_type = verified_fact.get('type', 'general')
        
        base_delta = 0.5
        
        if fact_type == 'physical':
            base_delta = 0.7
        elif fact_type == 'causal':
            base_delta = 0.8
        elif fact_type == 'mathematical':
            base_delta = 0.9
        elif fact_type == 'contradiction':
            base_delta = -0.5
        
        confidence = verified_fact.get('confidence', 0.9)
        
        return base_delta * confidence
    
    def update_from_feedback(self, verified_fact: Dict[str, Any], learning_rate: float = 0.015):
        """
        LLM验证通过的事实 → 更新几何流形
        
        Args:
            verified_fact: Dict with keys: 'summary', 'type', 'confidence', 'region'
            learning_rate: ρ更新强度
        """
        delta_rho = self._fact_to_geometric_delta(verified_fact)
        confidence = verified_fact.get('confidence', 0.9)
        
        if self.rho is None:
            self.rho = 0.5
        
        self.rho = (1 - learning_rate * confidence) * self.rho + \
                    learning_rate * confidence * delta_rho
        
        self.rho = max(0.1, min(0.99, self.rho))
        
        self.rho_history.append({
            'timestamp': np.datetime64('now'),
            'rho': self.rho,
            'fact_type': verified_fact.get('type'),
            'confidence': confidence
        })
        
        self.update_curvature_field()
        
        if confidence > 0.85 and 'region' in verified_fact:
            self.refine_topology(verified_fact['region'])
        
        print(f"[动态ρ更新] 已吸收事实: {verified_fact.get('summary', 'unknown')} | 置信度: {confidence} | ρ: {self.rho:.4f}")
    
    def update_curvature_field(self):
        """重新计算背景曲率场"""
        if self.coords is None or len(self.coords) == 0:
            return
        
        eta = self.rho if self.rho is not None else 1.0
        
        for i in range(len(self.coords)):
            if i == 0:
                avg_dist = 0.5
            else:
                distances = cdist(self.coords[i].reshape(1, -1), self.coords[:i], 'euclidean')
                avg_dist = np.mean(distances) if distances.size > 0 else 0.5
            
            self.R_bg[i] = self._compute_background_curvature(avg_dist, eta)
    
    def refine_topology(self, region: str):
        """触发局部拓扑重构"""
        if self.coords is None or len(self.coords) < 2:
            return
        
        region_hash = hash(region) % len(self.coords)
        target_idx = min(region_hash, len(self.coords) - 1)
        
        target_coord = self.coords[target_idx]
        distances = cdist(target_coord.reshape(1, -1), self.coords)[0]
        nearest_indices = np.argsort(distances)[1:4]
        
        for idx in nearest_indices:
            self.coords[idx] = 0.95 * self.coords[idx] + 0.05 * target_coord
            if idx < len(self.R_bg):
                self.R_bg[idx] = (self.R_bg[idx] + self.R_bg[target_idx]) / 2
    
    def get_scaled_manifold(self, complexity_score: float):
        """
        根据问题复杂度动态选择尺度
        
        Args:
            complexity_score: 0~1 (由LLM或规则评估)
        
        Returns:
            对应尺度的简化流形视图
        """
        if complexity_score < 0.3:
            scale = 1
        elif complexity_score < 0.6:
            scale = 2
        elif complexity_score < 0.85:
            scale = 4
        else:
            scale = 8
        
        scale_key = f"scale_{scale}"
        if scale_key in self.scaled_manifolds:
            return self.scaled_manifolds[scale_key]
        
        scaled_view = self._create_coarse_grained_view(scale)
        self.scaled_manifolds[scale_key] = scaled_view
        
        return scaled_view
    
    def _create_coarse_grained_view(self, scale: int) -> Dict[str, Any]:
        """创建粗粒化版本（降低计算量）"""
        if self.coords is None:
            return {
                'coords': None,
                'R_bg': None,
                'rho': self.rho,
                'scale': scale,
                'dim': self.dim // scale
            }
        
        new_dim = max(1, self.dim // scale)
        
        if scale == 1:
            return {
                'coords': self.coords,
                'R_bg': self.R_bg,
                'rho': self.rho,
                'scale': scale,
                'dim': self.dim
            }
        
        coords_reshaped = self.coords.reshape(len(self.coords), scale, -1)
        coords_pooled = np.mean(coords_reshaped, axis=1)
        
        if self.R_bg is not None and len(self.R_bg) >= scale:
            num_groups = len(self.R_bg) // scale
            if num_groups > 0:
                R_bg_trimmed = self.R_bg[:num_groups * scale]
                R_bg_reshaped = R_bg_trimmed.reshape(num_groups, scale)
                R_bg_pooled = np.mean(R_bg_reshaped, axis=1)
            else:
                R_bg_pooled = self.R_bg
        else:
            R_bg_pooled = self.R_bg
        
        return {
            'coords': coords_pooled,
            'R_bg': R_bg_pooled,
            'rho': self.rho,
            'scale': scale,
            'dim': new_dim
        }
    
    def get_rho_history(self) -> List[Dict[str, Any]]:
        return self.rho_history
    
    def _text_to_features(self, text: str) -> np.ndarray:
        char_counts = np.zeros(26)
        for char in text.lower():
            if 'a' <= char <= 'z':
                char_counts[ord(char) - ord('a')] += 1
        if np.sum(char_counts) > 0:
            char_counts = char_counts / np.sum(char_counts)
        
        words = text.split()
        word_len_dist = np.zeros(10)
        for word in words:
            idx = min(len(word) - 1, 9)
            word_len_dist[idx] += 1
        if np.sum(word_len_dist) > 0:
            word_len_dist = word_len_dist / np.sum(word_len_dist)
        
        features = np.concatenate([
            char_counts,
            word_len_dist,
            np.array([len(text), len(words), len(set(words)) / max(len(words), 1)])
        ])
        
        padding = max(0, self.dim - len(features))
        if padding > 0:
            features = np.concatenate([features, np.zeros(padding)])
        
        return features[:self.dim]
    
    def _multimodal_to_features(self, data: Any) -> np.ndarray:
        if hasattr(data, 'shape'):
            arr = np.array(data)
            flattened = arr.flatten()[:self.dim]
            if len(flattened) < self.dim:
                flattened = np.pad(flattened, (0, self.dim - len(flattened)))
            return flattened
        return np.array([hash(str(data)) % 1000 / 1000])

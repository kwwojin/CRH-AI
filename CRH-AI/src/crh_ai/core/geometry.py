import numpy as np
from typing import Dict, Any


class GeometricEngine:
    def __init__(self, c: float = 1.0):
        self.c = c  
        self.G = 6.67430e-11  
        self.scale_factor = 1.0
    
    def geometric_redshift(self, d: float, R_bg: float) -> float:
        return np.exp(R_bg * d) - 1
    
    def distance(self, pos1: np.ndarray, pos2: np.ndarray, R_bg: float = 0.0) -> float:
        diff = pos2 - pos1
        base_distance = np.linalg.norm(diff)
        
        if R_bg > 0:
            redshift_factor = self.geometric_redshift(base_distance, R_bg)
            return base_distance * (1 + redshift_factor)
        
        return base_distance
    
    def geodesic_distance(self, pos1: np.ndarray, pos2: np.ndarray, R_bg: float = 0.0) -> float:
        diff = pos2 - pos1
        
        if R_bg <= 0:
            return np.linalg.norm(diff)
        
        n = 100
        t = np.linspace(0, 1, n)
        points = pos1 + t[:, np.newaxis] * diff
        
        total_distance = 0.0
        for i in range(n - 1):
            segment = points[i + 1] - points[i]
            local_curvature = R_bg * (1 - t[i])
            redshift = self.geometric_redshift(np.linalg.norm(segment), local_curvature)
            total_distance += np.linalg.norm(segment) * (1 + redshift)
        
        return total_distance
    
    def causal_check(self, event1: Dict[str, Any], event2: Dict[str, Any]) -> bool:
        if 't' not in event1 or 't' not in event2:
            return True
        
        if 'x' not in event1 or 'x' not in event2:
            return True
        
        dt = event2['t'] - event1['t']
        dx = np.linalg.norm(event2['x'] - event1['x'])
        
        return dt >= dx / self.c
    
    def light_cone(self, event: Dict[str, Any], time_window: float = 1.0) -> Dict[str, np.ndarray]:
        if 't' not in event or 'x' not in event:
            return {'past': None, 'future': None}
        
        t0 = event['t']
        x0 = event['x']
        
        future_boundary = x0 + self.c * time_window
        past_boundary = x0 - self.c * time_window
        
        return {
            'past': np.array([[t0 - time_window, coord] for coord in past_boundary]),
            'future': np.array([[t0 + time_window, coord] for coord in future_boundary]),
            'apex': np.array([t0, *x0])
        }
    
    def parallel_transport(self, vector: np.ndarray, start_pos: np.ndarray, end_pos: np.ndarray, R_bg: float) -> np.ndarray:
        distance = self.distance(start_pos, end_pos, R_bg)
        
        if distance < 1e-10:
            return vector
        
        redshift = self.geometric_redshift(distance, R_bg)
        transported = vector * (1 + redshift)
        
        direction = (end_pos - start_pos) / distance
        dot_product = np.dot(vector, direction)
        transported -= dot_product * direction * (1 - 1/(1 + redshift))
        
        return transported
    
    def curvature_scalar(self, position: np.ndarray, manifold_coords: np.ndarray) -> float:
        if len(manifold_coords) < 4:
            return 0.0
        
        distances = np.linalg.norm(manifold_coords - position, axis=1)
        nearest_indices = np.argsort(distances)[:4]
        nearest_points = manifold_coords[nearest_indices]
        
        vectors = nearest_points - position
        gram_matrix = np.dot(vectors, vectors.T)
        
        try:
            det = np.linalg.det(gram_matrix)
            volume = np.sqrt(np.abs(det))
            return 1.0 / (volume + 1e-10)
        except np.linalg.LinAlgError:
            return 0.0

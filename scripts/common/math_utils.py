#!/usr/bin/env python3
"""공용 수학 유틸리티 함수."""

import numpy as np


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) → 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x**2 + y**2)],
    ], dtype=np.float64)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """벡터 배열을 단위 벡터로 정규화. (N, 3) 또는 (3,) 지원."""
    if vectors.size == 0:
        return vectors
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        return vectors / max(norm, 1e-9)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-9)

from typing import Any
import numpy as np
from scipy.special import softmax
class NAC_confidence:
    def __init__(self, k: int, s: float) -> None:
        """
        Implemetns Neighborhood Aware Cosine (NAC) that computes 
        similarity based on neighborhood information

        https://arxiv.org/pdf/2301.01922.pdf

        :param k: for kNN
        :param s: scale (=1/T)
        """
        self.k = k
        self.s = s
    def __call__(self, similarity_matrix: np.ndarray) -> Any:
        """
        :param similarity: (probe_size, gallery_size) marix, which specifies closeness of all test images to each gallery class
        :return probe_score: specifies confinence that particular test image belongs to predicted class
            image's probe_score is less than operating threshold τ, then this image get rejected as imposter
        """
        top_k_logits = np.sort(similarity_matrix, axis=1)[:, -self.k:]
        return softmax(top_k_logits * self.s, axis=1)[:, -1]
class MaxSimilarity_confidence:
    def __init__(self, foo: str) -> None:
        """
        Returns confidence for each test image of being gallery (known class) sample
        Here we take similarity to most similar class in gallery as confidence measure
        """
        self.foo = foo
    def __call__(self, similarity_matrix: np.ndarray) -> Any:
        """
        :param similarity: (probe_size, gallery_size) marix, which specifies closeness of all test images to each gallery class
        :return probe_score: specifies confinence that particular test image belongs to predicted class
            image's probe_score is less than operating threshold τ, then this image get rejected as imposter
        """
        return np.max(similarity_matrix, axis=1)
    

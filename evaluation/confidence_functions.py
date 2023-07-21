from typing import Any
import numpy as np
from scipy.special import softmax

from abc import ABC


class AbstractConfidence(ABC):
    def __call__(self, similarity_matrix) -> Any:
        raise NotImplementedError


class MisesProb(AbstractConfidence):
    def __init__(self, k: float, reject_prior: float) -> None:
        """
        Performes K+1 class classification, with K being number of gallery classed and
        K+1-th class is ood class.
        returns probabilities p(K+1|z) = \frac{p(z|c)p(c)}{p(z)} that test emb z belongs to reject class K+1
        where
        1. p(z|K+1) is uniform disribution on sphere
        2. p(z|c), c \leq K is von Mises-Fisher (vMF) distribution with koncentration k
        3. p(K+1) = reject_prior and p(1)= ... =p(K) = (1 - reject_prior)/K

        :param k: von Mises-Fisher (vMF) distribution with koncentration
        :param reject_prior: prior probability of ood sample, p(K+1)
        """
        self.k = k
        self.reject_prior = reject_prior

    def __call__(self, similarity_matrix: np.ndarray) -> Any:
        """
        :param similarity: (probe_size, gallery_size) marix, which specifies closeness of all test images to each gallery class
        :return probe_score: specifies confinence that particular test image belongs to predicted class
            image's probe_score is less than operating threshold τ, then this image get rejected as imposter
        """
        p_uniform = 0.1  # dencity of uniform distribution on sphere
        probe_score = similarity_matrix

        return probe_score


class NAC_confidence(AbstractConfidence):
    def __init__(self, k: int, s: float, normalize: bool) -> None:
        """
        Implemetns Neighborhood Aware Cosine (NAC) that computes
        similarity based on neighborhood information

        https://arxiv.org/pdf/2301.01922.pdf

        :param k: for kNN
        :param s: scale (=1/T)
        """
        self.k = k  # 15 is a good value
        self.s = s  # 1 is a good value
        self.normalize = normalize

    def __call__(self, similarity_matrix: np.ndarray) -> Any:
        """
        :param similarity: (probe_size, gallery_size) marix, which specifies closeness of all test images to each gallery class
        :return probe_score: specifies confinence that particular test image belongs to predicted class
            image's probe_score is less than operating threshold τ, then this image get rejected as imposter

        """
        if self.normalize:
            similarity_matrix = (
                similarity_matrix - np.mean(similarity_matrix, axis=1, keepdims=True)
            ) / np.std(similarity_matrix, axis=1, keepdims=True)
        top_k_logits = np.sort(similarity_matrix, axis=1)[:, -self.k :]

        return softmax((top_k_logits) * self.s, axis=1)[:, -1]


class MaxSimilarity_confidence(AbstractConfidence):
    """
    Returns confidence for each test image of being gallery (known class) sample
    Here we take similarity to most similar class in gallery as confidence measure
    """

    def __call__(self, similarity_matrix: np.ndarray) -> Any:
        """
        :param similarity: (probe_size, gallery_size) marix, which specifies closeness of all test images to each gallery class
        :return probe_score: specifies confinence that particular test image belongs to predicted class
            image's probe_score is less than operating threshold τ, then this image get rejected as imposter
        """
        return np.max(similarity_matrix, axis=1)

from typing import Any
import numpy as np
from scipy.special import softmax

from abc import ABC


class AbstractConfidence(ABC):
    def __call__(self, similarity_matrix) -> Any:
        raise NotImplementedError


from scipy.special import ive, hyp0f1, loggamma


class MisesProb(AbstractConfidence):
    def __init__(self, kappa: float, beta: float, d=512) -> None:
        """
        Performes K+1 class classification, with K being number of gallery classed and
        K+1-th class is ood class.
        returns probabilities p(K+1|z) = \frac{p(z|c)p(c)}{p(z)} that test emb z belongs to reject class K+1
        where
        1. p(z|K+1) is uniform disribution on sphere
        2. p(z|c), c \leq K is von Mises-Fisher (vMF) distribution with koncentration k
        3. p(K+1) = beta and p(1)= ... =p(K) = (1 - beta)/K

        :param kappa: koncentration for von Mises-Fisher (vMF) distribution
        :param beta: prior probability of ood sample, p(K+1)
        """
        self.kappa = kappa
        self.beta = beta
        self.n = d / 2
        self.alpha = hyp0f1(self.n, self.kappa**2 / 4, dtype=np.float64)
        self.log_iv = np.log(ive(self.n - 1, self.kappa, dtype=np.float64)) + self.kappa
        self.log_c = (
            (self.n - 1) * np.log(kappa) - self.n * np.log(2 * np.pi) - self.log_iv
        )
        self.log_uniform_dencity = (
            loggamma(self.n, dtype=np.float64) - np.log(2) - self.n * np.log(np.pi)
        )

    def __call__(self, similarity_matrix: np.ndarray) -> Any:
        """
        :param similarity: (probe_size, gallery_size) marix, which specifies closeness of all test images to each gallery class
        :return probe_score: specifies confinence that particular test image belongs to predicted class
            image's probe_score is less than operating threshold τ, then this image get rejected as imposter
        """

        return -self.compute_uniform_log_probability(similarity_matrix)

    def compute_log_z_prob(self, similarities: np.ndarray):
        K = similarities.shape[-1]

        logit_sum = (
            np.sum(np.exp(similarities * self.kappa), axis=-1) * (1 - self.beta) / K
        )
        # print(f'Logit sum: {logit_sum}')

        # print(f'Alpha value: {alpha_value}')
        log_z_prob = self.log_c + np.log(logit_sum + self.alpha * self.beta)
        # print(f'Log z prob: {log_z_prob}')
        return log_z_prob

    def compute_all_class_log_probabilities(self, similarities: np.ndarray):
        uniform_log_prob = self.compute_uniform_log_probability(similarities)

        # compute gallery classes log prob
        K = similarities.shape[-1]
        log_z_prob = self.compute_log_z_prob(similarities)
        gallery_log_probs = (
            self.log_c
            + self.kappa * similarities
            + np.log((1 - self.beta) / K)
            - log_z_prob[..., np.newaxis]
        )

        return np.concatenate(
            [gallery_log_probs, uniform_log_prob[..., np.newaxis]], axis=-1
        )

    def compute_uniform_log_probability(self, similarities: np.ndarray):
        # compute log z prob
        log_z_prob = self.compute_log_z_prob(similarities)
        # print(f'Log z prob: {log_z_prob}')

        # print(f'Log uniform dencity: {log_uniform_dencity}')
        log_beta = np.log(self.beta)
        # print(f'Log beta : {log_beta}')
        log_prob = self.log_uniform_dencity + log_beta - log_z_prob
        # print(f'Log uniform prob: {log_prob}')
        return log_prob


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
        return np.max(similarity_matrix, axis=-1)
